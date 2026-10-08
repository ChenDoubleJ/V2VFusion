# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers
import xformers.ops
from einops import rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.vision_transformer import Mlp
import torch as th

USE_TEMPORAL_TRANSFORMER = True


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate="tanh"), token_num=120):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0
        )
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class DropPath(nn.Module):
    r"""DropPath but without rescaling and supports optional all-zero and/or all-keep.
    """

    def __init__(self, p):
        super(DropPath, self).__init__()
        self.p = p

    def forward(self, *args, zero=None, keep=None):
        if not self.training:
            return args[0] if len(args) == 1 else args

        # params
        x = args[0]
        b = x.size(0)
        n = (torch.rand(b) < self.p).sum()

        # non-zero and non-keep mask
        mask = x.new_ones(b, dtype=torch.bool)
        if keep is not None:
            mask[keep] = False
        if zero is not None:
            mask[zero] = False

        # drop-path index
        index = torch.where(mask)[0]
        index = index[torch.randperm(len(index))[:n]]
        if zero is not None:
            index = torch.cat([index, torch.where(zero)[0]], dim=0)

        # drop-path multiplier
        multiplier = x.new_ones(b)
        multiplier[index] = 0.0
        output = tuple(u * self.broadcast(multiplier, u) for u in args)
        return output[0] if len(args) == 1 else output

    def broadcast(self, src, dst):
        assert src.size(0) == dst.size(0)
        shape = (dst.size(0), ) + (1, ) * (dst.ndim - 1)
        return src.view(shape)


def sinusoidal_embedding(timesteps, dim):
    # check input
    half = dim // 2
    timesteps = timesteps.float()

    # compute sinusoidal embedding
    sinusoid = torch.outer(
        timesteps, torch.pow(10000,
                             -torch.arange(half).to(timesteps).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    if dim % 2 != 0:
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    return x


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        mask = torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
        # aviod mask all, which will cause find_unused_parameters error
        if mask.all():
            mask[0] = False
        return mask


class MemoryEfficientCrossAttention(nn.Module):

    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 max_bs=16384,
                 dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.max_bs = max_bs
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3).reshape(b, t.shape[
                1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(
                    b * self.heads, t.shape[1], self.dim_head).contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of.
        if q.shape[0] > self.max_bs:
            q_list = torch.chunk(q, q.shape[0] // self.max_bs, dim=0)
            k_list = torch.chunk(k, k.shape[0] // self.max_bs, dim=0)
            v_list = torch.chunk(v, v.shape[0] // self.max_bs, dim=0)
            out_list = []
            for q_1, k_1, v_1 in zip(q_list, k_list, v_list):
                out = xformers.ops.memory_efficient_attention(
                    q_1, k_1, v_1, attn_bias=None, op=self.attention_op)
                out_list.append(out)
            out = torch.cat(out_list, dim=0)
        else:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0).reshape(
                b, self.heads, out.shape[1],
                self.dim_head).permute(0, 2, 1,
                                       3).reshape(b, out.shape[1],
                                                  self.heads * self.dim_head))
        return self.to_out(out)


class RelativePositionBias(nn.Module):

    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  num_buckets=32,
                                  max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact) *  # noqa
            (num_buckets - max_exact)).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 disable_self_attn=False,
                 use_linear=False,
                 use_checkpoint=True,
                 is_ctrl=False):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim[d],
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                local_type='space',
                is_ctrl=is_ctrl) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(
                    inner_dim, in_channels, kernel_size=1, stride=1,
                    padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        _, _, h, w = x.shape
        # print('x shape:', x.shape)  # [64, 320, 90, 160]
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], h=h, w=w)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


_ATTN_PRECISION = os.environ.get('ATTN_PRECISION', 'fp32')


class CrossAttention(nn.Module):

    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                      (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == 'fp32':
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        weight = torch.cat([max_out, avg_out], dim=1)
        weight = self.conv1(weight)

        out = self.sigmoid(weight) * x
        return out

class TemporalLocalAttention(nn.Module):
    def __init__(self):
        super(TemporalLocalAttention, self).__init__()
        self.conv1 = nn.Linear(in_features=2, out_features=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_out, _ = torch.max(x, dim=-1, keepdim=True)
        avg_out = torch.mean(x, dim=-1, keepdim=True)

        weight = torch.cat([max_out, avg_out], dim=-1)
        weight = self.conv1(weight)

        out = self.sigmoid(weight) * x
        return out


class BasicTransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_heads,
                 d_head,
                 dropout=0.,
                 context_dim=None,
                 gated_ff=True,
                 checkpoint=True,
                 disable_self_attn=False,
                 local_type=None,
                 is_ctrl=False):
        super().__init__()
        self.local_type = local_type
        self.is_ctrl = is_ctrl
        attn_cls = MemoryEfficientCrossAttention
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(  # self-attn
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        attn_cls2 = MemoryEfficientCrossAttention
            
        self.attn2 = attn_cls2(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

        if self.local_type == 'space' and self.is_ctrl:
            self.local1 = SpatialAttention()

        if self.local_type == 'temp' and self.is_ctrl:
            self.local1 = TemporalLocalAttention()
            self.local2 = TemporalLocalAttention()

    def forward_(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(),
                          self.checkpoint)

    def forward(self, x, context=None, h=None, w=None):

        if self.local_type == 'space' and self.is_ctrl:  # [b*t,(hw), c]

            x_local = rearrange(x, 'b (h w) c -> b c h w', h=h)
            x_local = self.local1(x_local)
            x_local = rearrange(x_local, 'b c h w -> b (h w) c')

            x = self.attn1(
            self.norm1(x_local),
            context=context if self.disable_self_attn else None) + x

            x = self.attn2(self.norm2(x), context=context) + x  # cross attention or self-attention
            x = self.ff(self.norm3(x)) + x

        if self.local_type == 'temp' and self.is_ctrl:

            x_local = self.local1(x)

            x = self.attn1(
            self.norm1(x_local),
            context=context if self.disable_self_attn else None) + x

            x_local = self.local2(x)

            x = self.attn2(self.norm2(x_local), context=context) + x
            x = self.ff(self.norm3(x)) + x

        return x


# feedforward
class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(
            dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout),
                                 nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(
                self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = x[..., 1:-1, :]
        if self.use_conv:
            x = self.conv(x)
        return x
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torch
# import torch.nn as nn
# import torch as th


class MoEBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        dims=2,
        use_temporal_conv=True,
        use_image_dataset=False,
        num_experts=8,
        topk_task=6,
        topk_deg=3,
    ):
        super().__init__()

        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.num_experts = num_experts
        self.topk_task = topk_task
        self.topk_deg = topk_deg
        self.use_temporal_conv = use_temporal_conv
        # ---------------- Base ----------------

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )

        # ---------------- Experts ----------------

        self.experts = nn.ModuleList([
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            for _ in range(num_experts)
        ])

        # ---------------- Routers ----------------

        self.task_gate = nn.Conv2d(self.out_channels, num_experts, 1)
        self.deg_gate = nn.Conv2d(self.out_channels, num_experts, 1)
        if self.use_temporal_conv:
            self.temporal_conv = TemporalConvBlock_v2(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                use_image_dataset=use_image_dataset,
            )
        self.context_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1024, self.out_channels)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb, hint, context, batch_size, variant_info=None):

        h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # # ---------- context ----------
        if context is not None:

            B, C, H, W = h.shape

            context_flat = context.mean(dim=1)

            context_out = self.context_layers(context_flat).type(h.dtype)

            while len(context_out.shape) < len(h.shape):
                context_out = context_out[..., None]

            context_out = context_out.expand(-1, -1, H, W)

        else:

            context_out = 0

        # ---------- routing ----------
# -------------
        # task_logits = self.task_gate(h)
        # # print(f'task_logits.shape: {task_logits.shape}')  # 
        # topk_task_val, topk_task_idx = th.topk(
        #     task_logits,
        #     self.topk_task,
        #     dim=1
        # )
        # # print(f'topk_task_idx.shape: {topk_task_idx.shape}')  #

        # task_mask = th.zeros_like(task_logits)

        # task_mask.scatter_(1, topk_task_idx, 1.0)

        # task_probs = th.sigmoid(task_logits)

        # task_mask = (task_mask - task_probs).detach() + task_probs

        # h2 = h + context_out
        # # h2 = h


        # deg_logits = self.deg_gate(h2)
        # # print(f'deg_logits.shape: {deg_logits.shape}')  #
        # deg_logits = deg_logits * task_mask
# -------------
        task_logits = self.task_gate(h)

        topk_task_val, topk_task_idx = th.topk(
            task_logits,
            self.topk_task,
            dim=1
        )

        # Hard mask（真正用于约束第二层）
        hard_task_mask = th.zeros_like(task_logits)
        hard_task_mask.scatter_(1, topk_task_idx, 1.0)

        # STE mask（仅用于梯度传播）
        task_probs = th.sigmoid(task_logits)
        task_mask = (hard_task_mask - task_probs).detach() + task_probs

        # ---------- 第二层 routing ----------

        h2 = h + context_out

        deg_logits = self.deg_gate(h2)

        
        deg_logits = deg_logits.masked_fill(hard_task_mask == 0, -1e4)
#--------------------------
        topk_deg_val, topk_deg_idx = th.topk(
            deg_logits,
            self.topk_deg,
            dim=1
        )
        # print(f'topk_deg_idx.shape: {topk_deg_idx.shape}')  #
        deg_mask = th.zeros_like(deg_logits)

        deg_mask.scatter_(1, topk_deg_idx, 1.0)

        gate_logits = deg_logits.masked_fill(deg_mask == 0, -1e4)

        gate = th.softmax(gate_logits, dim=1)
        # aa = gate.mean(dim=(0, 2, 3))/
        # print(aa)
        # ---------- experts ----------
       
        expert_outputs = []

        for expert in self.experts:
            out = expert(h)
            expert_outputs.append(out)

        expert_outputs = th.stack(expert_outputs, dim=1)

        gate = gate.unsqueeze(2)
        # print(gate.shape)

        h = (expert_outputs * gate).sum(dim=1)

        # ---------- resblock ----------

        h = h + emb_out

        h = self.out_layers(h)

        h = self.skip_connection(x) + h
        # ---------------- Temporal Conv ----------------

        if self.use_temporal_conv:

            h = rearrange(h, '(b f) c h w -> b c f h w', b=batch_size)

            h = self.temporal_conv(
                h,
                variant_info=variant_info
            )

            h = rearrange(h, 'b c f h w -> (b f) c h w')

        return h
# class MoEBlock(nn.Module):
#     def __init__(
#         self,
#         channels,
#         emb_channels,
#         dropout,
#         out_channels=None,
#         use_scale_shift_norm=False,
#         dims=2,
#         use_temporal_conv=True,
#         use_image_dataset=False,
#         num_experts=6,
#         topk_task=4,
#         topk_deg=2,
#     ):
#         super().__init__()

#         self.channels = channels
#         self.emb_channels = emb_channels
#         self.dropout = dropout
#         self.out_channels = out_channels or channels
#         self.use_scale_shift_norm = use_scale_shift_norm
#         self.use_temporal_conv = use_temporal_conv

#         self.num_experts = num_experts
#         self.topk_task = topk_task
#         self.topk_deg = topk_deg

#         # ---------------- Base ResBlock ----------------

#         self.in_layers = nn.Sequential(
#             nn.GroupNorm(32, channels),
#             nn.SiLU(),
#             nn.Conv2d(channels, self.out_channels, 3, padding=1),
#         )
#         self.emb_layers = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(emb_channels, self.out_channels),
#         )

#         self.out_layers = nn.Sequential(
#             nn.GroupNorm(32, self.out_channels),
#             nn.SiLU(),
#             nn.Dropout(p=dropout),
#             nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
#         )

#         # ---------------- Experts (标准 MoE) ----------------

#         self.experts = nn.ModuleList([
#             nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
#             for _ in range(num_experts)
#         ])

#         # ---------------- Task Router ----------------

#         self.task_gate = nn.Conv2d(
#             self.out_channels,
#             num_experts,
#             kernel_size=1
#         )

#         # ---------------- Degradation Router ----------------

#         self.deg_gate = nn.Conv2d(
#             self.out_channels,
#             num_experts,
#             kernel_size=1
#         )

#         
#         self.context_layers = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(1024, self.out_channels)
#         )

#         # ---------------- Skip ----------------

#         if self.out_channels == channels:
#             self.skip_connection = nn.Identity()
#         else:
#             self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

#         # ---------------- Temporal Conv ----------------

#         if self.use_temporal_conv:
#             self.temporal_conv = TemporalConvBlock_v2(
#                 self.out_channels,
#                 self.out_channels,
#                 dropout=0.1,
#                 use_image_dataset=use_image_dataset,
#             )

#     def forward(self, x, emb, hint, context, batch_size, variant_info=None):
#         return self._forward(x, emb, hint, context, batch_size, variant_info)
#     def _forward(self, x, emb, hint, context, batch_size, variant_info):

#         # ---------------- ResBlock ----------------
#         h = self.in_layers(x)
#         emb_out = self.emb_layers(emb).type(h.dtype)
#         while len(emb_out.shape) < len(h.shape):
#             emb_out = emb_out[..., None]

#         # 
#         if context is not None:
#             B, C, H, W = h.shape
#             context_flat = context.mean(dim=1)
#             context_out = self.context_layers(context_flat).type(h.dtype)  # [B, C]
#             while len(context_out.shape) < len(h.shape):
#                 context_out = context_out[..., None]
#             context_out = context_out.expand(-1, -1, H, W)

#         # ---------------- Hierarchical Routing ----------------

#         # ---------- Step1 Task Routing ----------
#         task_logits = self.task_gate(h)
        
#         topk_task_val, topk_task_idx = th.topk(
#             task_logits,
#             self.topk_task,
#             dim=1
#         )

#         task_mask = th.zeros_like(task_logits)
#         task_mask.scatter_(1, topk_task_idx, 1.0)

#         # 
#        
#         task_probs = th.sigmoid(task_logits) 
#         task_mask = (task_mask - task_probs).detach() + task_probs 

#         

#        
#         h2 = h + context_out
#         deg_logits = self.deg_gate(h2)

#         deg_logits = deg_logits * task_mask

#         topk_deg_val, topk_deg_idx = th.topk(
#             deg_logits,
#             self.topk_deg,
#             dim=1
#         )

#         deg_mask = th.zeros_like(deg_logits)
#         deg_mask.scatter_(1, topk_deg_idx, 1.0)

#         
#         gate_logits = deg_logits.masked_fill(deg_mask == 0, float('-inf'))
#         gate = th.softmax(gate_logits, dim=1)
#         gate = gate + 1e-6
#         gate = gate / gate.sum(dim=1, keepdim=True)
#         # ---------------- Experts Forward ----------------

#         expert_outputs = []

#         
#         for expert in self.experts:
#             out = expert(h) 
#             expert_outputs.append(out)

#         expert_outputs = th.stack(expert_outputs, dim=1)

#         # ---------------- Expert Weighted Sum ----------------

#         gate = gate.unsqueeze(2)

#         
#         h = (expert_outputs * gate).sum(dim=1)

#         # ---------------- Residual ----------------

#         h = h + emb_out
#         h = self.out_layers(h)

#         h = self.skip_connection(x) + h

        # ---------------- Temporal Conv ----------------

        # if self.use_temporal_conv:

        #     h = rearrange(h, '(b f) c h w -> b c f h w', b=batch_size)

        #     h = self.temporal_conv(
        #         h,
        #         variant_info=variant_info
        #     )

        #     h = rearrange(h, 'b c f h w -> (b f) c h w')

        # return h
    # def _forward(self, x, emb, hint, context, batch_size, variant_info):

    #     # ---------------- ResBlock ----------------
    #     # x.shape: torch.Size([30, 320, 90, 160])
    #     # hint.shape: torch.Size([30, 320, 90, 160])
    #     # context.shape: torch.Size([30, 77, 1024])
    #     # h.shape: torch.Size([30, 320, 90, 160])
    #     # print(f'x.shape: {x.shape}')
    #     # print(f'hint.shape: {hint.shape}')
    #     # print(f'context.shape: {context.shape}')
    #     h = self.in_layers(x)
    #     # print(f'h.shape after in_layers: {h.shape}')
    #     # print(f'h.shape: {h.shape}')
    #     emb_out = self.emb_layers(emb).type(h.dtype)
    #     while len(emb_out.shape) < len(h.shape):
    #         emb_out = emb_out[..., None]

    #     # 
    #     if context is not None:

    #         B, C, H, W = h.shape

    #         # 1. 展平 token 维度 [B, 77, 1024] -> [B, 77*1024]
    #         context_flat = context.reshape(context.shape[0], -1)

    #         # 
    #         context_out = self.context_layers(context_flat).type(h.dtype)  # [B, C]

    #         # 
    #         while len(context_out.shape) < len(h.shape):
    #             context_out = context_out[..., None]

    #         # 4. expand 到 feature map
    #         context_out = context_out.expand(-1, -1, H, W)
    #    # # --------------------------------------------------------------------------

    #     # ---------------- Hierarchical Routing ----------------

    #     # ---------- Step1 Task Routing ----------
    #     # h1 = h + hint
    #     task_logits = self.task_gate(h)

    #     topk_task_val, topk_task_idx = th.topk(
    #         task_logits,
    #         self.topk_task,
    #         dim=1
    #     )

    #     task_mask = th.zeros_like(task_logits)
    #     task_mask.scatter_(1, topk_task_idx, 1.0)

    #     # ---------- Step2 Degradation Routing ----------

    #   
    #     h2 = h + context_out
    #     deg_logits = self.deg_gate(h2)

    #     deg_logits = deg_logits * task_mask

    #     topk_deg_val, topk_deg_idx = th.topk(
    #         deg_logits,
    #         self.topk_deg,
    #         dim=1
    #     )

    #     deg_mask = th.zeros_like(deg_logits)
    #     deg_mask.scatter_(1, topk_deg_idx, 1.0)

    #     gate_logits = deg_logits * deg_mask

    #     gate = th.softmax(gate_logits, dim=1)

    #     # ---------------- Experts Forward ----------------

    #     expert_outputs = []

    #     for expert in self.experts:
    #         out = expert(h)
    #         expert_outputs.append(out)

    #     expert_outputs = th.stack(expert_outputs, dim=1)
    #     # shape
    #     # B E C H W

    #     # ---------------- Expert Weighted Sum ----------------

    #     gate = gate.unsqueeze(2)

    #     h = (expert_outputs * gate).sum(dim=1)

    #     # ---------------- Residual ----------------

    #     h = h + emb_out
    #     h = self.out_layers(h)

    #     h = self.skip_connection(x) + h

    #     # ---------------- Temporal Conv ----------------

    #     if self.use_temporal_conv:

    #         h = rearrange(h, '(b f) c h w -> b c f h w', b=batch_size)

    #         h = self.temporal_conv(
    #             h,
    #             variant_info=variant_info
    #         )

    #         h = rearrange(h, 'b c f h w -> (b f) c h w')

    #     return h

# class MoEBlock(nn.Module):
#     def __init__(
#         self,
#         channels,
#         emb_channels,
#         dropout,
#         out_channels=None,
#         use_scale_shift_norm=False,
#         dims=2,
#         use_temporal_conv=True,
#         use_image_dataset=False,
#         num_experts=8,
#         topk=4,
#     ):
#         super().__init__()

#         self.channels = channels
#         self.emb_channels = emb_channels
#         self.dropout = dropout
#         self.out_channels = out_channels or channels
#         self.use_scale_shift_norm = use_scale_shift_norm
#         self.use_temporal_conv = use_temporal_conv

#         self.num_experts = num_experts
#         self.topk = topk

#         assert self.out_channels % num_experts == 0

#         # ---------------- Base ResBlock ----------------

#         self.in_layers = nn.Sequential(
#             nn.GroupNorm(32, channels),
#             nn.SiLU(),
#             nn.Conv2d(channels, self.out_channels, 3, padding=1),
#         )

#         self.emb_layers = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(emb_channels, self.out_channels),
#         )

#         self.out_layers = nn.Sequential(
#             nn.GroupNorm(32, self.out_channels),
#             nn.SiLU(),
#             nn.Dropout(p=dropout),
#             nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
#         )

#         # ---------------- Task Router ----------------

#         self.task_gate = nn.Conv2d(
#             self.out_channels,
#             num_experts,
#             kernel_size=1
#         )

#         # ---------------- Degradation Router ----------------

#         self.deg_gate = nn.Conv2d(
#             self.out_channels,
#             num_experts,
#             kernel_size=1
#         )

#         # ---------------- Skip ----------------

#         if self.out_channels == channels:
#             self.skip_connection = nn.Identity()
#         else:
#             self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

#         # ---------------- Temporal Conv ----------------

#         if self.use_temporal_conv:
#             self.temporal_conv = TemporalConvBlock_v2(
#                 self.out_channels,
#                 self.out_channels,
#                 dropout=0.1,
#                 use_image_dataset=use_image_dataset,
#             )

#     def forward(self, x, emb, batch_size, variant_info=None):
#         return self._forward(x, emb, batch_size, variant_info)

#     def _forward(self, x, emb, batch_size, variant_info):

#         # ---------------- ResBlock ----------------

#         h = self.in_layers(x)

#         emb_out = self.emb_layers(emb).type(h.dtype)
#         while len(emb_out.shape) < len(h.shape):
#             emb_out = emb_out[..., None]

#         h = h + emb_out
#         h = self.out_layers(h)

#         # ---------------- Dual Routing ----------------

#         task_logits = self.task_gate(h)
#         deg_logits = self.deg_gate(h)

#         gate_logits = task_logits + deg_logits

#         # ---------------- Top-K Routing ----------------

#         topk_val, topk_idx = th.topk(
#             gate_logits,
#             self.topk,
#             dim=1
#         )

#         mask = th.zeros_like(gate_logits)
#         mask.scatter_(1, topk_idx, 1.0)

#         gate_logits = gate_logits * mask

#         gate = th.softmax(gate_logits, dim=1)

#         # ---------------- Expert Mixing ----------------

#         chunks = th.chunk(h, self.num_experts, dim=1)

#         mixed = []
#         for i, c in enumerate(chunks):

#             g = gate[:, i:i+1]

#             mixed.append(c * g)

#         h = th.cat(mixed, dim=1)

#         # ---------------- Residual ----------------

#         h = self.skip_connection(x) + h

#         # ---------------- Temporal Conv ----------------

#         if self.use_temporal_conv:

#             h = rearrange(h, '(b f) c h w -> b c f h w', b=batch_size)

#             h = self.temporal_conv(
#                 h,
#                 variant_info=variant_info
#             )

#             h = rearrange(h, 'b c f h w -> (b f) c h w')

#         return h
class ResmoeBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        use_temporal_conv=True,
        use_image_dataset=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels
                if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock_v2(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                use_image_dataset=use_image_dataset)

    def forward(self, x, emb, hint, context, batch_size, variant_info=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :param hint: an [N x hint_channels] Tensor of hint embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb, hint, context, batch_size, variant_info)

    def _forward(self, x, emb, hint, context, batch_size, variant_info):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x) 
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:  # false
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv:
            h = rearrange(h, '(b f) c h w -> b c f h w', b=batch_size)
            h = self.temopral_conv(h, variant_info=variant_info)
            h = rearrange(h, 'b c f h w -> (b f) c h w')
        return h

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        use_temporal_conv=True,
        use_image_dataset=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels
                if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock_v2(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                use_image_dataset=use_image_dataset)

    def forward(self, x, emb, batch_size, variant_info=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb, batch_size, variant_info)

    def _forward(self, x, emb, batch_size, variant_info):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:  # false
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv:
            h = rearrange(h, '(b f) c h w -> b c f h w', b=batch_size)
            h = self.temopral_conv(h, variant_info=variant_info)
            h = rearrange(h, 'b c f h w -> (b f) c h w')
        return h


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=(2, 1)):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2d(
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Resample(nn.Module):

    def __init__(self, in_dim, out_dim, mode):
        assert mode in ['none', 'upsample', 'downsample']
        super(Resample, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mode = mode

    def forward(self, x, reference=None):
        if self.mode == 'upsample':
            assert reference is not None
            x = F.interpolate(x, size=reference.shape[-2:], mode='nearest')
        elif self.mode == 'downsample':
            x = F.adaptive_avg_pool2d(
                x, output_size=tuple(u // 2 for u in x.shape[-2:]))
        return x


class ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 embed_dim,
                 out_dim,
                 use_scale_shift_norm=True,
                 mode='none',
                 dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.use_scale_shift_norm = use_scale_shift_norm
        self.mode = mode

        # layers
        self.layer1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1))
        self.resample = Resample(in_dim, in_dim, mode)
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim,
                      out_dim * 2 if use_scale_shift_norm else out_dim))
        self.layer2 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1))
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Conv2d(
            in_dim, out_dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.layer2[-1].weight)

    def forward(self, x, e, reference=None):
        identity = self.resample(x, reference)
        x = self.layer1[-1](self.resample(self.layer1[:-1](x), reference))
        e = self.embedding(e).unsqueeze(-1).unsqueeze(-1).type(x.dtype)
        if self.use_scale_shift_norm:
            scale, shift = e.chunk(2, dim=1)
            x = self.layer2[0](x) * (1 + scale) + shift
            x = self.layer2[1:](x)
        else:
            x = x + e
            x = self.layer2(x)
        x = x + self.shortcut(identity)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim, context_dim=None, num_heads=None, head_dim=None):
        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.pow(head_dim, -0.25)

        # layers
        self.norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        if context_dim is not None:
            self.context_kv = nn.Linear(context_dim, dim * 2)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x, context=None):
        r"""x:       [B, C, H, W].
            context: [B, L, C] or None.
        """
        identity = x
        b, c, h, w, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        x = self.norm(x)
        q, k, v = self.to_qkv(x).view(b, n * 3, d, h * w).chunk(3, dim=1)
        if context is not None:
            ck, cv = self.context_kv(context).reshape(b, -1, n * 2,
                                                      d).permute(0, 2, 3,
                                                                 1).chunk(
                                                                     2, dim=1)
            k = torch.cat([ck, k], dim=-1)
            v = torch.cat([cv, v], dim=-1)

        # compute attention
        attn = torch.matmul(q.transpose(-1, -2) * self.scale, k * self.scale)
        attn = F.softmax(attn, dim=-1)

        # gather context
        x = torch.matmul(v, attn.transpose(-1, -2))
        x = x.reshape(b, c, h, w)

        # output
        x = self.proj(x)
        return x + identity


class TemporalAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 heads=4,
                 dim_head=32,
                 rotary_emb=None,
                 use_image_dataset=False,
                 use_sim_mask=False):
        super().__init__()
        # consider num_heads first, as pos_bias needs fixed num_heads
        dim_head = dim // heads
        assert heads * dim_head == dim
        self.use_image_dataset = use_image_dataset
        self.use_sim_mask = use_sim_mask

        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = nn.GroupNorm(32, dim)
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self,
                x,
                pos_bias=None,
                focus_present_mask=None,
                video_mask=None):

        identity = x
        n, height, device = x.shape[2], x.shape[-2], x.device

        x = self.norm(x)
        x = rearrange(x, 'b c f h w -> b (h w) f c')

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values （v=qkv[-1]） through to the output
            values = qkv[-1]
            out = self.to_out(values)
            out = rearrange(out, 'b (h w) f c -> b c f h w', h=height)

            return out + identity

        # split out heads
        q = rearrange(qkv[0], '... n (h d) -> ... h n d', h=self.heads)
        k = rearrange(qkv[1], '... n (h d) -> ... h n d', h=self.heads)
        v = rearrange(qkv[2], '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity
        # shape [b (hw) h n n], n=f
        sim = torch.einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if (focus_present_mask is None and video_mask is not None):
            # video_mask: [B, n]
            mask = video_mask[:, None, :] * video_mask[:, :, None]
            mask = mask.unsqueeze(1).unsqueeze(1)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        elif exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n),
                                         device=device,
                                         dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.use_sim_mask:
            sim_mask = torch.tril(
                torch.ones((n, n), device=device, dtype=torch.bool),
                diagonal=0)
            sim = sim.masked_fill(~sim_mask, -torch.finfo(sim.dtype).max)

        # numerical stability
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = torch.einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        out = self.to_out(out)

        out = rearrange(out, 'b (h w) f c -> b c f h w', h=height)

        if self.use_image_dataset:
            out = identity + 0 * out
        else:
            out = identity + out
        return out


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 disable_self_attn=False,
                 use_linear=False,
                 use_checkpoint=True,
                 only_self_att=True,
                 multiply_zero=False,
                 is_ctrl=False):
        super().__init__()
        self.multiply_zero = multiply_zero
        self.only_self_att = only_self_att
        self.use_adaptor = False
        if self.only_self_att:
            context_dim = None
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv1d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)
            if self.use_adaptor:
                self.adaptor_in = nn.Linear(frames, frames)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim[d],
                checkpoint=use_checkpoint,
                local_type='temp',
                is_ctrl=is_ctrl) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(
                    inner_dim, in_channels, kernel_size=1, stride=1,
                    padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
            if self.use_adaptor:
                self.adaptor_out = nn.Linear(frames, frames)
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if self.only_self_att:
            context = None
        if not isinstance(context, list):
            context = [context]
        b, _, _, h, w = x.shape
        x_in = x
        x = self.norm(x)

        if not self.use_linear:
            x = rearrange(x, 'b c f h w -> (b h w) c f').contiguous()
            x = self.proj_in(x)
        if self.use_linear:
            x = rearrange(
                x, 'b c f h w -> (b h w) f c').contiguous()
            x = self.proj_in(x)
            x = rearrange(
                x, 'bhw f c -> bhw c f').contiguous()

        # print('x shape:', x.shape)  # [28800, 512, 32]
        if self.only_self_att:  # no cross-attention
            x = rearrange(x, 'bhw c f -> bhw f c').contiguous()
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, h=h, w=w)
            # print('x shape:', x.shape)  # [43200, 32, 512]
            x = rearrange(x, '(b hw) f c -> b hw f c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) c f -> b hw f c', b=b).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                context[i] = rearrange(
                    context[i], '(b f) l con -> b f l con',
                    f=self.frames).contiguous()
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_i_j = repeat(
                        context[i][j],
                        'f l con -> (f r) l con',
                        r=(h * w) // self.frames,
                        f=self.frames).contiguous()
                    x[j] = block(x[j], context=context_i_j)

        if self.use_linear:
            x = rearrange(x, 'b hw f c -> (b hw) f c').contiguous()
            x = self.proj_out(x)
            x = rearrange(
                x, '(b h w) f c -> b c f h w', b=b, h=h, w=w).contiguous()
        if not self.use_linear:
            # print('x shape:', x.shape)  # [2, 21600, 32, 512]
            x = rearrange(x, 'b hw f c -> (b hw) c f').contiguous()
            x = self.proj_out(x)
            x = rearrange(
                x, '(b h w) c f -> b c f h w', b=b, h=h, w=w).contiguous()

        if self.multiply_zero:
            x = 0.0 * x + x_in
        else:
            x = x + x_in
        return x


class TemporalAttentionMultiBlock(nn.Module):

    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None,
        use_image_dataset=False,
        use_sim_mask=False,
        temporal_attn_times=1,
    ):
        super().__init__()
        self.att_layers = nn.ModuleList([
            TemporalAttentionBlock(dim, heads, dim_head, rotary_emb,
                                   use_image_dataset, use_sim_mask)
            for _ in range(temporal_attn_times)
        ])

    def forward(self,
                x,
                pos_bias=None,
                focus_present_mask=None,
                video_mask=None):
        for layer in self.att_layers:
            x = layer(x, pos_bias, focus_present_mask, video_mask)
        return x


class InitTemporalConvBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim=None,
                 dropout=0.0,
                 use_image_dataset=False):
        super(InitTemporalConvBlock, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        if self.use_image_dataset:
            x = identity + 0 * x
        else:
            x = identity + x
        return x


class TemporalConvBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim=None,
                 dropout=0.0,
                 use_image_dataset=False):
        super(TemporalConvBlock, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv2[-1].weight)
        nn.init.zeros_(self.conv2[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_image_dataset:
            x = identity + 0 * x
        else:
            x = identity + x
        return x


class TemporalConvBlock_v2(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim=None,
                 dropout=0.0,
                 use_image_dataset=False):
        super(TemporalConvBlock_v2, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, x, variant_info=None):
        if variant_info is not None and variant_info.get('type') == 'variant2':
            # print(x.shape)  # torch.Size([1, 320, 32, 90, 160])
            _, _, f, _, _ = x.shape
            assert f % 4 == 0, "f must be divisible by 4"
            x_short = rearrange(x, "b c (n s) h w -> (n b) c s h w", n=4)
            x_short = self.conv1(x_short)
            x_short = self.conv2(x_short)
            x_short = self.conv3(x_short)
            x_short = self.conv4(x_short)
            x_short = rearrange(x_short, "(n b) c s h w -> b c (n s) h w", n=4)

            identity = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)

            x = x * (1-variant_info['alpha']) + x_short * variant_info['alpha']
        

        elif variant_info is not None and variant_info.get('type') == 'variant1':
            identity = x
            x_long, x_short = x.chunk(2, dim=0)

            x_short = rearrange(x_short, "b c (n s) h w -> (n b) c s h w", n=4)
            x_short = self.conv1(x_short)
            x_short = self.conv2(x_short)
            x_short = self.conv3(x_short)
            x_short = self.conv4(x_short)
            x_short = rearrange(x_short, "(n b) c s h w -> b c (n s) h w", n=4)

            x_long = self.conv1(x_long)
            x_long = self.conv2(x_long)
            x_long = self.conv3(x_long)
            x_long = self.conv4(x_long)

            x = torch.cat([x_long, x_short], dim=0)
        

        elif variant_info is None:
            identity = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            

        if self.use_image_dataset:
            x = identity + 0.0 * x
        else:
            x = identity + x
        return x


class Vid2VidSDUNet(nn.Module):

    def __init__(self,
                 in_dim=4,
                 dim=320,
                 y_dim=1024,
                 context_dim=1024,
                 out_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_heads=8,
                 head_dim=64,
                 num_res_blocks=2,
                 attn_scales=[1 / 1, 1 / 2, 1 / 4],
                 use_scale_shift_norm=True,
                 dropout=0.1,
                 temporal_attn_times=1,
                 temporal_attention=True,
                 use_checkpoint=True,
                 use_image_dataset=False,
                 use_fps_condition=False,
                 use_sim_mask=False,
                 training=False,
                 inpainting=True):
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        super(Vid2VidSDUNet, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        # for temporal attention
        self.num_heads = num_heads
        # for spatial attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_fps_condition = use_fps_condition
        self.use_sim_mask = use_sim_mask
        self.training = training
        self.inpainting = inpainting

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))

        if self.use_fps_condition:
            self.fps_embedding = nn.Sequential(
                nn.Linear(dim, embed_dim), nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)

        # encoder
        self.input_blocks = nn.ModuleList()
        init_block = nn.ModuleList([nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        # need an initial temporal attention?
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(
                    TemporalTransformer(
                        dim,
                        num_heads,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                        is_ctrl=True
                        ))
            else:
                init_block.append(
                    TemporalAttentionMultiBlock(
                        dim,
                        num_heads,
                        head_dim,
                        rotary_emb=self.rotary_emb,
                        temporal_attn_times=temporal_attn_times,
                        use_image_dataset=use_image_dataset))
        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim,
                out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.ModuleList([
                    ResBlock(
                        in_dim,
                        embed_dim,
                        dropout,
                        out_channels=out_dim,
                        use_scale_shift_norm=False,
                        use_image_dataset=use_image_dataset,
                    )
                ])
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=self.context_dim,
                            disable_self_attn=False,
                            use_linear=True,
                            is_ctrl=True
                            ))
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset,
                                    is_ctrl=True
                                    ))
                        else:
                            block.append(
                                TemporalAttentionMultiBlock(
                                    out_dim,
                                    num_heads,
                                    head_dim,
                                    rotary_emb=self.rotary_emb,
                                    use_image_dataset=use_image_dataset,
                                    use_sim_mask=use_sim_mask,
                                    temporal_attn_times=temporal_attn_times))
                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(
                        out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)

        self.middle_block = nn.ModuleList([
            ResBlock(
                out_dim,
                embed_dim,
                dropout,
                use_scale_shift_norm=False,
                use_image_dataset=use_image_dataset,
            ),
            SpatialTransformer(
                out_dim,
                out_dim // head_dim,
                head_dim,
                depth=1,
                context_dim=self.context_dim,
                disable_self_attn=False,
                use_linear=True,
                is_ctrl=True
                )
        ])

        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
                    TemporalTransformer(
                        out_dim,
                        out_dim // head_dim,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                        is_ctrl=True

                    ))
            else:
                self.middle_block.append(
                    TemporalAttentionMultiBlock(
                        out_dim,
                        num_heads,
                        head_dim,
                        rotary_emb=self.rotary_emb,
                        use_image_dataset=use_image_dataset,
                        use_sim_mask=use_sim_mask,
                        temporal_attn_times=temporal_attn_times))

        self.middle_block.append(
            ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))

        # decoder
        self.output_blocks = nn.ModuleList()
        for i, (in_dim,
                out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                block = nn.ModuleList([
                    ResBlock(
                        in_dim + shortcut_dims.pop(),
                        embed_dim,
                        dropout,
                        out_dim,
                        use_scale_shift_norm=False,
                        use_image_dataset=use_image_dataset,
                    )
                ])
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=1024,
                            disable_self_attn=False,
                            use_linear=True,
                            is_ctrl=True))
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset,
                                    is_ctrl=True))
                        else:
                            block.append(
                                TemporalAttentionMultiBlock(
                                    out_dim,
                                    num_heads,
                                    head_dim,
                                    rotary_emb=self.rotary_emb,
                                    use_image_dataset=use_image_dataset,
                                    use_sim_mask=use_sim_mask,
                                    temporal_attn_times=temporal_attn_times))
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = Upsample(
                        out_dim, True, dims=2.0, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                self.output_blocks.append(block)

        # head
        self.out = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))

        # zero out the last layer params
        nn.init.zeros_(self.out[-1].weight)

    def forward(self,
                x,
                t,
                y,
                x_lr=None,
                fps=None,
                video_mask=None,
                focus_present_mask=None,
                prob_focus_present=0.,
                mask_last_frame_num=0):

        batch, c, f, h, w = x.shape
        device = x.device
        self.batch = batch

        # image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(
                focus_present_mask, lambda: prob_mask_like(
                    (batch, ), prob_focus_present, device=device))

        if self.temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            time_rel_pos_bias = self.time_rel_pos_bias(
                x.shape[2], device=x.device)
        else:
            time_rel_pos_bias = None

        # embeddings
        e = self.time_embed(sinusoidal_embedding(t, self.dim))
        context = y

        # repeat f times for spatial e and context
        e = e.repeat_interleave(repeats=f, dim=0)
        context = context.repeat_interleave(repeats=f, dim=0)

        # always in shape (b f) c h w, except for temporal layer
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        # encoder
        xs = []
        for ind, block in enumerate(self.input_blocks):
            x = self._forward_single(block, x, e, context, time_rel_pos_bias,
                                     focus_present_mask, video_mask)
            xs.append(x)

        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias,
                                     focus_present_mask, video_mask)

        # decoder
        for block in self.output_blocks:
            x = torch.cat([x, xs.pop()], dim=1)
            x = self._forward_single(
                block,
                x,
                e,
                context,
                time_rel_pos_bias,
                focus_present_mask,
                video_mask,
                reference=xs[-1] if len(xs) > 0 else None)

        # head
        x = self.out(x)

        # reshape back to (b c f h w)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=batch)
        return x

    def _forward_single(self,
                        module,
                        x,
                        e,
                        context,
                        time_rel_pos_bias,
                        focus_present_mask,
                        video_mask,
                        reference=None):
        if isinstance(module, ResidualBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, reference)
        # elif isinstance(module, MoEBlock):
        #     module = checkpoint_wrapper(
        #         module) if self.use_checkpoint else module
        #     x = x.contiguous()
        #     x = module(x, e, context, self.batch)
        elif isinstance(module, ResBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, context)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, CrossAttention):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, MemoryEfficientCrossAttention):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, FeedForward):
            x = module(x, context)
        elif isinstance(module, Upsample):
            x = module(x)
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, Resample):
            x = module(x, reference)
        elif isinstance(module, TemporalAttentionBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalAttentionMultiBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, InitTemporalConvBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalConvBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, e, context,
                                         time_rel_pos_bias, focus_present_mask,
                                         video_mask, reference)
        else:
            x = module(x)
        return x


class ControlledV2VUNet(Vid2VidSDUNet):
    def __init__(self):
        super(ControlledV2VUNet, self).__init__()
        self.VideoControlNet = VideoControlNet()

    def forward(self,
                x,
                t,
                y,
                hint=None,
                variant_info=None,
                hint_chunk=None,
                t_hint=None,
                s_cond=None,
                mask_cond=None,
                x_lr=None,
                fps=None,
                mask=None,
                video_mask=None,
                focus_present_mask=None,
                prob_focus_present=0.,
                mask_last_frame_num=0,
                ):
        
        batch, _, f, _, _= x.shape
        device = x.device
        self.batch = batch

        # Process text (new added for t5 encoder)
        # y = self.VideoControlNet.y_embedder(y, self.training).squeeze(1)  # [1, 1, 120, 4096] -> [B, 1, 120, 1024].squeeze(1) -> [B, 120, 1024]

        if hint_chunk is not None:
            hint = hint_chunk

        control = self.VideoControlNet(x, t, y, hint=hint, t_hint=t_hint, \
                                                mask_cond=mask_cond, s_cond=s_cond, \
                                                variant_info=variant_info)
                                                
        # image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(
                focus_present_mask, lambda: prob_mask_like(
                    (batch, ), prob_focus_present, device=device))

        if self.temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            time_rel_pos_bias = self.time_rel_pos_bias(
                x.shape[2], device=x.device)
        else:
            time_rel_pos_bias = None

        e = self.time_embed(sinusoidal_embedding(t, self.dim)) 
        e = e.repeat_interleave(repeats=f, dim=0)

        # context = y
        context = y.repeat_interleave(repeats=f, dim=0)

        # always in shape (b f) c h w, except for temporal layer
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        # encoder
        xs = []
        for block in self.input_blocks:
            x = self._forward_single(block, x, e, hint, context, time_rel_pos_bias,
                                    focus_present_mask, video_mask, variant_info=variant_info)
            xs.append(x)
        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, e, hint, context, time_rel_pos_bias,
                                    focus_present_mask, video_mask, variant_info=variant_info)
            
        if control is not None:
            x = control.pop() + x

        # decoder
        for block in self.output_blocks:
            if control is None:
                x = torch.cat([x, xs.pop()], dim=1)
                # print(x.size())
            else:
                # print(f'The size of x is: {x.size()}')
                # print(f'The size of xs.pop(): {xs[-1].size()}')
                # print(f'The size of control.pop() is: {control[-1].size()}')
                # print((xs.pop()+control.pop()).size())
                # print(control.pop().size())
                # a = xs.pop()
                # b = control.pop()
                x = torch.cat([x, xs.pop() + control.pop()], dim=1)
                # x = torch.cat([x, a+b], dim=1)

            x = self._forward_single(
                block,
                x,
                e,
                hint,
                context,
                time_rel_pos_bias,
                focus_present_mask,
                video_mask,
                reference=xs[-1] if len(xs) > 0 else None,
                variant_info=variant_info)

        # head
        x = self.out(x)

        # reshape back to (b c f h w)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=batch)
        return x

    def _forward_single(self,
                        module,
                        x,
                        e,
                        hint,
                        context,
                        time_rel_pos_bias,
                        focus_present_mask,
                        video_mask,
                        reference=None,
                        variant_info=None):
        variant_info = None # For Debug
        if isinstance(module, ResidualBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, MoEBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, hint, context, self.batch, variant_info)
        elif isinstance(module, ResBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, self.batch, variant_info)
        elif isinstance(module, SpatialTransformer):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, context)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, CrossAttention):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, MemoryEfficientCrossAttention):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, FeedForward):
            x = module(x, context)
        elif isinstance(module, Upsample):
            x = module(x)
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, Resample):
            x = module(x, reference)
        elif isinstance(module, TemporalAttentionBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalAttentionMultiBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, InitTemporalConvBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalConvBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, e, hint, context,
                                         time_rel_pos_bias, focus_present_mask,
                                         video_mask, reference, variant_info)
        else:
            x = module(x)
        return x


class VideoControlNet(nn.Module):

    def __init__(self,
                 in_dim=4,
                 dim=320,
                 y_dim=1024,
                 context_dim=1024,
                 out_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_heads=8,
                 head_dim=64,
                 num_res_blocks=2,
                 attn_scales=[1 / 1, 1 / 2, 1 / 4],
                 use_scale_shift_norm=True,
                 dropout=0.1,
                 temporal_attn_times=1,
                 temporal_attention=True,
                 use_checkpoint=True,
                 use_image_dataset=False,
                 use_fps_condition=False,
                 use_sim_mask=False,
                 training=False,
                 inpainting=True):
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        super(VideoControlNet, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        # for temporal attention
        self.num_heads = num_heads
        # for spatial attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_fps_condition = use_fps_condition
        self.use_sim_mask = use_sim_mask
        self.training = training
        self.inpainting = inpainting

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        
        if self.use_fps_condition:
            self.fps_embedding = nn.Sequential(
                nn.Linear(dim, embed_dim), nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)

        # encoder
        self.input_blocks = nn.ModuleList()
        init_block = nn.ModuleList([nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        # need an initial temporal attention?
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(
                    TemporalTransformer(
                        dim,
                        num_heads,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                        is_ctrl=True,))
            else:
                init_block.append(
                    TemporalAttentionMultiBlock(
                        dim,
                        num_heads,
                        head_dim,
                        rotary_emb=self.rotary_emb,
                        temporal_attn_times=temporal_attn_times,
                        use_image_dataset=use_image_dataset))
        self.input_blocks.append(init_block)
        self.zero_convs = nn.ModuleList([self.make_zero_conv(dim)])
        shortcut_dims.append(dim)
        for i, (in_dim,
                out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.ModuleList([
                    MoEBlock(
                        in_dim,
                        embed_dim,
                        dropout,
                        out_channels=out_dim,
                        use_scale_shift_norm=False,
                        use_image_dataset=use_image_dataset,
                    )
                ])
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=self.context_dim,
                            disable_self_attn=False,
                            use_linear=True,
                            is_ctrl=True))
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset,
                                    is_ctrl=True,))
                        else:
                            block.append(
                                TemporalAttentionMultiBlock(
                                    out_dim,
                                    num_heads,
                                    head_dim,
                                    rotary_emb=self.rotary_emb,
                                    use_image_dataset=use_image_dataset,
                                    use_sim_mask=use_sim_mask,
                                    temporal_attn_times=temporal_attn_times))
                in_dim = out_dim
                self.input_blocks.append(block)
                self.zero_convs.append(self.make_zero_conv(out_dim)) 
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(
                        out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)
                    self.zero_convs.append(self.make_zero_conv(out_dim)) 

        self.middle_block = nn.ModuleList([
            MoEBlock(
                out_dim,
                embed_dim,
                dropout,
                use_scale_shift_norm=False,
                use_image_dataset=use_image_dataset,
            ),
            SpatialTransformer(
                out_dim,
                out_dim // head_dim,
                head_dim,
                depth=1,
                context_dim=self.context_dim,
                disable_self_attn=False,
                use_linear=True,
                is_ctrl=True)
        ])

        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
                    TemporalTransformer(
                        out_dim,
                        out_dim // head_dim,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                        is_ctrl=True,
                    ))
            else:
                self.middle_block.append(
                    TemporalAttentionMultiBlock(
                        out_dim,
                        num_heads,
                        head_dim,
                        rotary_emb=self.rotary_emb,
                        use_image_dataset=use_image_dataset,
                        use_sim_mask=use_sim_mask,
                        temporal_attn_times=temporal_attn_times))

        self.middle_block.append(
            ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))
        
        self.middle_block_out = self.make_zero_conv(embed_dim)

        '''
        add prompt
        '''
        add_dim = 320
        self.add_dim = add_dim

        self.input_hint_block = zero_module(nn.Conv2d(4, add_dim, 3, padding=1))

    def make_zero_conv(self, in_channels, out_channels=None):
        out_channels = in_channels if out_channels is None else out_channels
        return TimestepEmbedSequential(zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0)))

    def forward(self,
                x,
                t,
                y,
                s_cond=None,
                hint=None,
                variant_info=None,
                t_hint=None,
                mask_cond=None,
                fps=None,
                video_mask=None,
                focus_present_mask=None,
                prob_focus_present=0.,
                mask_last_frame_num=0):

        batch, _, f, _, _ = x.shape
        device = x.device
        self.batch = batch
        # print(y)
        # image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(
                focus_present_mask, lambda: prob_mask_like(
                    (batch, ), prob_focus_present, device=device))

        if self.temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            time_rel_pos_bias = self.time_rel_pos_bias(
                x.shape[2], device=x.device)
        else:
            time_rel_pos_bias = None

        if hint is not None:
            # print('hint0 shape:', hint.shape)  # [1, 4, 30, 90, 160]
            # add = x.new_zeros(batch, self.add_dim, f, h, w)
            hint = rearrange(hint, 'b c f h w -> (b f) c h w')
            hint = self.input_hint_block(hint)
            # print('hint shape after input_hint_block:', hint.shape)  # [30, 320, 90, 160]
            # hint = rearrange(hint, '(b f) c h w -> b c f h w', b = batch)
        # print('e0 shape:', e.shape)
        e = self.time_embed(sinusoidal_embedding(t, self.dim)) # [1, 1280]
        # print('e1 shape:', e.shape)sss
        e = e.repeat_interleave(repeats=f, dim=0)  # 时间编码 # [30, 1280]
        # print('e2 shape:', e.shape)
        # print('y0 shape:', y.shape)
        context = y.repeat_interleave(repeats=f, dim=0)  # 文本内容 [30, 77, 1024]
        # text = context
        # print('y shape:', context.shape)
        # print('context shape:', context.shape) 
        # always in shape (b f) c h w, except for temporal layern 
        x = rearrange(x, 'b c f h w -> (b f) c h w')  # 输入视频
        # print('before x shape:', x.shape) [30, 320, 90, 160]
        # print('hint shape:', hint.shape) [30, 320, 90, 160]

        # encoder
        xs = []
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if hint is not None:
                hint1 = hint
                # print('hint is not None')
                for block in module:
                    x = self._forward_single(block, x, e, hint1, context, time_rel_pos_bias,
                                        focus_present_mask, video_mask, variant_info=variant_info)
                    if not isinstance(block, TemporalTransformer):
                        if hint is not None:
                            # print(f'hint.shape: {hint.shape}')
                            x += hint
                            hint = None
            else:
                x = self._forward_single(module, x, e, hint1,context, time_rel_pos_bias,
                                        focus_present_mask, video_mask, variant_info=variant_info)
            xs.append(zero_conv(x, e, context))

        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, e, hint1, context, time_rel_pos_bias,
                                     focus_present_mask, video_mask, variant_info=variant_info)
        xs.append(self.middle_block_out(x, e, context))
       
        return xs

    def _forward_single(self,
                        module,
                        x,
                        e,
                        hint,
                        context,
                        time_rel_pos_bias,
                        focus_present_mask,
                        video_mask,
                        reference=None,
                        variant_info=None,):
        # variant_info = None # For Debug
        if isinstance(module, ResidualBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, MoEBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = x.contiguous()
            # hint = hint.contiguous()
            # context = context.contiguous()
            x = module(x, e, hint, context, self.batch, variant_info)
        elif isinstance(module, ResBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, self.batch, variant_info)
        elif isinstance(module, SpatialTransformer):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            # print("x shape:", x.shape)  # [2, 320, 32, 90, 160]
            x = module(x, context)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, CrossAttention):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, MemoryEfficientCrossAttention):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, FeedForward):
            x = module(x, context)
        elif isinstance(module, Upsample):
            x = module(x)
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, Resample):
            x = module(x, reference)
        elif isinstance(module, TemporalAttentionBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalAttentionMultiBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, InitTemporalConvBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalConvBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, e, hint, context,
                                         time_rel_pos_bias, focus_present_mask,
                                         video_mask, reference, variant_info)
        else:
            x = module(x)
        return x


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x
