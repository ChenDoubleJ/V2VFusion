import os
import torch
from argparse import ArgumentParser, Namespace
import json
from typing import Any, Dict, List, Mapping, Tuple
from easydict import EasyDict

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(base_path)
from video_to_video.video_to_video_model import VideoToVideo_fusion
from video_to_video.utils.seed import setup_seed
from video_to_video.utils.logger import get_logger
from video_fusion.color_fix import adain_color_fix
from video_fusion.dataset import TestPairedCaptionVideoDataset
from inference_utils import *
from tqdm.auto import tqdm
logger = get_logger()


class V2VF():
    def __init__(self, 
                 result_dir='./results/',
                 model_path='',
                 solver_mode='fast',
                 steps=1,
                 guide_scale=7.5,
                 max_chunk_len=32,
                 ):
        self.model_path=model_path
        logger.info('checkpoint_path: {}'.format(self.model_path))

        self.result_dir = result_dir
        # self.file_name = file_name
        os.makedirs(self.result_dir, exist_ok=True)

        model_cfg = EasyDict(__name__='model_cfg')
        model_cfg.model_path = self.model_path
        self.model = VideoToVideo_fusion(model_cfg)

        steps = 1 if solver_mode == 'fast' else steps
        self.solver_mode=solver_mode
        self.steps=steps
        self.guide_scale=guide_scale
        # self.upscale = upscale
        self.max_chunk_len=max_chunk_len

    def enhance_a_video(self, test_dataloader):
        for batch in tqdm(test_dataloader):
            video_data0 = batch.pop("lq")
            video_data10 = batch.pop("lq1")
            text = batch.pop("text")[0]
            file_name = batch.pop("name")[0]
            fps = self.max_chunk_len
            video_data0 = video_data0.squeeze(0)
            video_data10 = video_data10.squeeze(0)
            # print(video_data0.shape)
            # print(video_data10.shape)
            # print(text)
            # file_name = file_name.squeeze(0)
            # text = text.squeeze(0)
            # fps = fps.squeeze(0)
            # logger.info('input video path: {}'.format(video_path))
            # logger.info('input video1 path: {}'.format(video_path1))
            # text = prompt
            # text1 = prompt1
            logger.info('text: {}'.format(text))
            logger.info('name: {}'.format(file_name))
            # logger.info('text1: {}'.format(text1))
            # logger.info('positive_prompt: {}'.format(self.model.positive_prompt))
            # caption = text + self.model.positive_prompt
            caption = text

            # input_frames, input_fps = load_video(video_path)
            # input_frames1, input_fps1 = load_video(video_path1)
            # logger.info('input fps: {}'.format(input_fps))
            # logger.info('input fps1: {}'.format(input_fps1))
            logger.info('input fps: {}'.format(fps))

    #         video_data0 = preprocess(input_frames)
    #         video_data10 = preprocess(input_frames1)
    # # ***
            _, _, h0, w0 = video_data0.shape
            video_data = torch.nn.functional.interpolate(
                video_data0, size=(720, 1280), mode='bilinear', align_corners=False
            )
            video_data1 = torch.nn.functional.interpolate(
                video_data10, size=(720, 1280), mode='bilinear', align_corners=False
            )
    # 888
            _, _, h, w = video_data.shape
            # print(video_data.shape)
            logger.info('input resolution: {}'.format((h, w)))
            # target_h, target_w = h * self.upscale, w * self.upscale   # adjust_resolution(h, w, up_scale=4)
            # logger.info('target resolution: {}'.format((target_h, target_w)))

            pre_data = {'video_data': video_data, 'y': caption, 'video_data1': video_data1}
            # pre_data['target_res'] = (target_h, target_w)
            # pre_data['target_res'] = (h, w)


            total_noise_levels = 900
            setup_seed(666)

            with torch.no_grad():
                data_tensor = collate_fn(pre_data, 'cuda:0')
                output = self.model.test(data_tensor, total_noise_levels, steps=self.steps, \
                                    solver_mode=self.solver_mode, guide_scale=self.guide_scale, \
                                    max_chunk_len=self.max_chunk_len
                                    )

            output = tensor2vid(output)
            
            output = output.permute(0, 3, 1, 2)
            output = torch.nn.functional.interpolate(
                output, size=(h0, w0), mode='bilinear', align_corners=False
            )
            output = output.permute(0, 2, 3, 1)
            # Using color fix
            # output = adain_color_fix(output, video_data0)
            save_video(output, self.result_dir, file_name, fps=fps)
            os.path.join(self.result_dir, file_name)
        

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--input_path", required=True, type=str, help="input video path")
    # parser.add_argument("--input_path1", required=True, type=str, help="input video path")
    parser.add_argument("--save_dir", type=str, default='results', help="save directory")
    # parser.add_argument("--file_name", type=str, help="file name")
    parser.add_argument("--model_path", type=str, default='./pretrained_weight/model.pt', help="model path")
    # parser.add_argument("--prompt", type=str, default='a good video', help="prompt")
    # parser.add_argument("--prompt1", type=str, default='a good video', help="prompt")
    parser.add_argument("--upscale", type=int, default=1, help='up-scale')
    parser.add_argument("--max_chunk_len", type=int, default=30, help='max_chunk_len')

    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--solver_mode", type=str, default='fast', help='fast | normal')
    parser.add_argument("--steps", type=int, default=1)

    return parser.parse_args()

def main():
    
    args = parse_args()

    input_path = args.input_path
    # prompt = args.prompt
    # input_path1 = args.input_path1
    model_path = args.model_path
    save_dir = args.save_dir
    # file_name = args.file_name
    # upscale = args.upscale
    max_chunk_len = args.max_chunk_len

    steps = args.steps
    solver_mode = args.solver_mode
    guide_scale = args.cfg

    assert solver_mode in ('fast', 'normal')
    test_dataset = TestPairedCaptionVideoDataset(
    root_folders=[
        input_path
    ], 
    num_frames=max_chunk_len,)

    test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    num_workers=1,
    batch_size=1,
    shuffle=False)

    star = V2VF(
                result_dir=save_dir,
                model_path=model_path,
                solver_mode=solver_mode,
                steps=steps,
                guide_scale=guide_scale,
                max_chunk_len=max_chunk_len,
                )

    star.enhance_a_video(test_dataloader)

if __name__ == '__main__':
    main()


# import os
# import sys
# import json
# import torch

# from argparse import ArgumentParser
# from typing import Any, Dict, List, Mapping, Tuple

# from easydict import EasyDict
# from tqdm.auto import tqdm
# from torch.profiler import profile, ProfilerActivity


# # -------------------------------------------------------------------------
# # Project path
# # -------------------------------------------------------------------------
# base_path = os.path.abspath(
#     os.path.join(
#         os.path.dirname(__file__),
#         "../../",
#     )
# )
# sys.path.append(base_path)


# # -------------------------------------------------------------------------
# # Project imports
# # -------------------------------------------------------------------------
# from video_to_video.video_to_video_model import VideoToVideo_fusion
# from video_to_video.utils.seed import setup_seed
# from video_to_video.utils.logger import get_logger
# from video_fusion.color_fix import adain_color_fix
# from video_fusion.dataset import TestPairedCaptionVideoDataset
# from inference_utils import *


# logger = get_logger()


# class STAR:
#     def __init__(
#         self,
#         result_dir="./results/",
#         model_path="",
#         solver_mode="fast",
#         steps=1,
#         guide_scale=7.5,
#         max_chunk_len=32,
#         profile_flops=False,
#         profile_detail=False,
#         profile_trace="",
#         profile_only=False,
#     ):
#         self.model_path = model_path

#         logger.info(
#             "checkpoint_path: {}".format(
#                 self.model_path
#             )
#         )

#         self.result_dir = result_dir
#         os.makedirs(
#             self.result_dir,
#             exist_ok=True,
#         )

#         # -------------------------------------------------------------
#         # Build model
#         # -------------------------------------------------------------
#         model_cfg = EasyDict(
#             __name__="model_cfg"
#         )
#         model_cfg.model_path = self.model_path

#         self.model = VideoToVideo_fusion(
#             model_cfg
#         )

#         # fast 模式固定一步
#         self.steps = (
#             1
#             if solver_mode == "fast"
#             else steps
#         )

#         self.solver_mode = solver_mode
#         self.guide_scale = guide_scale
#         self.max_chunk_len = max_chunk_len

#         # -------------------------------------------------------------
#         # Profiler settings
#         # -------------------------------------------------------------
#         self.profile_flops = profile_flops
#         self.profile_detail = profile_detail
#         self.profile_trace = profile_trace
#         self.profile_only = profile_only

#         # 只统计第一个 batch
#         self._has_profiled = False

#         self.print_parameter_count()

#     def print_parameter_count(self):
#         """
#         输出模型参数量。

#         注意：
#         这里统计 self.model 中注册的全部参数，可能包含训练阶段使用、
#         但当前融合推理路径未调用的模块。
#         """
#         if not hasattr(
#             self.model,
#             "parameters",
#         ):
#             logger.warning(
#                 "self.model has no parameters() method; "
#                 "parameter count is unavailable."
#             )
#             return

#         try:
#             total_params = sum(
#                 parameter.numel()
#                 for parameter in self.model.parameters()
#             )

#             trainable_params = sum(
#                 parameter.numel()
#                 for parameter in self.model.parameters()
#                 if parameter.requires_grad
#             )

#             logger.info(
#                 "Total parameters: {:,} ({:.4f} M)".format(
#                     total_params,
#                     total_params / 1e6,
#                 )
#             )

#             logger.info(
#                 "Trainable parameters: {:,} ({:.4f} M)".format(
#                     trainable_params,
#                     trainable_params / 1e6,
#                 )
#             )

#         except Exception as error:
#             logger.warning(
#                 "Failed to count model parameters: {}".format(
#                     error
#                 )
#             )

#     def run_model_test(
#         self,
#         data_tensor,
#         total_noise_levels,
#     ):
#         """
#         执行完整的视频融合推理。

#         正常推理和 FLOPs 统计都调用这个函数，
#         确保两者使用完全相同的推理路径。
#         """
#         output = self.model.test(
#             data_tensor,
#             total_noise_levels,
#             steps=self.steps,
#             solver_mode=self.solver_mode,
#             guide_scale=self.guide_scale,
#             max_chunk_len=self.max_chunk_len,
#         )

#         return output

#     @staticmethod
#     def get_profile_total_flops(prof):
#         """
#         汇总 torch.profiler 已识别算子的 FLOPs。

#         使用 key_averages()，其中相同算子的多次调用已经聚合。
#         """
#         total_flops = 0.0
#         operators_with_flops = []

#         for event in prof.key_averages():
#             event_flops = getattr(
#                 event,
#                 "flops",
#                 None,
#             )

#             if event_flops is None:
#                 continue

#             event_flops = float(
#                 event_flops
#             )

#             if event_flops <= 0:
#                 continue

#             total_flops += event_flops

#             operators_with_flops.append(
#                 (
#                     event.key,
#                     event_flops,
#                     event.count,
#                 )
#             )

#         operators_with_flops.sort(
#             key=lambda item: item[1],
#             reverse=True,
#         )

#         return (
#             total_flops,
#             operators_with_flops,
#         )

#     def profile_model_test(
#         self,
#         data_tensor,
#         total_noise_levels,
#         num_frames,
#         height,
#         width,
#     ):
#         """
#         统计一次完整 self.model.test() 的 FLOPs。

#         输出：
#             1. 整个视频片段 GFLOPs
#             2. 平均每帧 GFLOPs
#             3. 平均每采样步 GFLOPs
#             4. 可选的算子明细
#             5. 可选的 Chrome Trace
#         """
#         activities = [
#             ProfilerActivity.CPU
#         ]

#         if torch.cuda.is_available():
#             activities.append(
#                 ProfilerActivity.CUDA
#             )

#             # 等待统计前已有 CUDA 任务执行完毕
#             torch.cuda.synchronize()

#         logger.info(
#             "=" * 80
#         )
#         logger.info(
#             "Starting FLOPs profiling"
#         )
#         logger.info(
#             "Frames: {}".format(
#                 num_frames
#             )
#         )
#         logger.info(
#             "Resolution: {} x {}".format(
#                 height,
#                 width,
#             )
#         )
#         logger.info(
#             "Solver mode: {}".format(
#                 self.solver_mode
#             )
#         )
#         logger.info(
#             "Sampling steps: {}".format(
#                 self.steps
#             )
#         )
#         logger.info(
#             "Guidance scale: {}".format(
#                 self.guide_scale
#             )
#         )
#         logger.info(
#             "Maximum chunk length: {}".format(
#                 self.max_chunk_len
#             )
#         )

#         with torch.no_grad():
#             with profile(
#                 activities=activities,
#                 record_shapes=True,
#                 profile_memory=False,
#                 with_flops=True,
#                 with_stack=False,
#             ) as prof:

#                 output = self.run_model_test(
#                     data_tensor=data_tensor,
#                     total_noise_levels=total_noise_levels,
#                 )

#                 # CUDA 为异步执行，必须同步后再退出 profiler
#                 if torch.cuda.is_available():
#                     torch.cuda.synchronize()

#         (
#             total_flops,
#             operators_with_flops,
#         ) = self.get_profile_total_flops(
#             prof
#         )

#         total_gflops = (
#             total_flops / 1e9
#         )

#         effective_frames = max(
#             int(num_frames),
#             1,
#         )

#         effective_steps = max(
#             int(self.steps),
#             1,
#         )

#         gflops_per_frame = (
#             total_gflops
#             / effective_frames
#         )

#         gflops_per_step = (
#             total_gflops
#             / effective_steps
#         )

#         gflops_per_frame_per_step = (
#             total_gflops
#             / effective_frames
#             / effective_steps
#         )

#         logger.info(
#             "-" * 80
#         )
#         logger.info(
#             "Recognized FLOPs: {:.6e}".format(
#                 total_flops
#             )
#         )
#         logger.info(
#             "GFLOPs / clip: {:.4f}".format(
#                 total_gflops
#             )
#         )
#         logger.info(
#             "GFLOPs / frame: {:.4f}".format(
#                 gflops_per_frame
#             )
#         )
#         logger.info(
#             "GFLOPs / sampling step: {:.4f}".format(
#                 gflops_per_step
#             )
#         )
#         logger.info(
#             "GFLOPs / frame / sampling step: {:.4f}".format(
#                 gflops_per_frame_per_step
#             )
#         )

#         if total_flops == 0:
#             logger.warning(
#                 "No FLOPs were recognized by torch.profiler. "
#                 "The model may mainly use unsupported operators "
#                 "or custom CUDA kernels."
#             )

#         # -------------------------------------------------------------
#         # Print operators that contribute FLOPs
#         # -------------------------------------------------------------
#         if operators_with_flops:
#             logger.info(
#                 "-" * 80
#             )
#             logger.info(
#                 "Top operators by recognized FLOPs:"
#             )

#             for (
#                 operator_name,
#                 operator_flops,
#                 operator_count,
#             ) in operators_with_flops[:30]:

#                 logger.info(
#                     "{:<40s} {:>12.4f} GFLOPs, calls={}".format(
#                         operator_name,
#                         operator_flops / 1e9,
#                         operator_count,
#                     )
#                 )

#         # -------------------------------------------------------------
#         # Full profiler table
#         # -------------------------------------------------------------
#         if self.profile_detail:
#             logger.info(
#                 "-" * 80
#             )
#             logger.info(
#                 "Detailed profiler table:"
#             )

#             if torch.cuda.is_available():
#                 sort_key = (
#                     "self_cuda_time_total"
#                 )
#             else:
#                 sort_key = (
#                     "self_cpu_time_total"
#                 )

#             print(
#                 prof.key_averages(
#                     group_by_input_shape=True
#                 ).table(
#                     sort_by=sort_key,
#                     row_limit=100,
#                 )
#             )

#         # -------------------------------------------------------------
#         # Save Chrome Trace
#         # -------------------------------------------------------------
#         if self.profile_trace:
#             trace_path = os.path.abspath(
#                 self.profile_trace
#             )

#             trace_directory = os.path.dirname(
#                 trace_path
#             )

#             if trace_directory:
#                 os.makedirs(
#                     trace_directory,
#                     exist_ok=True,
#                 )

#             prof.export_chrome_trace(
#                 trace_path
#             )

#             logger.info(
#                 "Profiler trace saved to: {}".format(
#                     trace_path
#                 )
#             )

#         logger.info(
#             "=" * 80
#         )

#         return output

#     def enhance_a_video(
#         self,
#         test_dataloader,
#     ):
#         for batch_index, batch in enumerate(
#             tqdm(test_dataloader)
#         ):
#             # ---------------------------------------------------------
#             # Load batch
#             # ---------------------------------------------------------
#             video_data0 = batch.pop(
#                 "lq"
#             )
#             video_data10 = batch.pop(
#                 "lq1"
#             )
#             text = batch.pop(
#                 "text"
#             )[0]
#             file_name = batch.pop(
#                 "name"
#             )[0]

#             # 保留你的原始设置
#             fps = self.max_chunk_len

#             # DataLoader batch size 为 1
#             # [1, T, C, H, W] -> [T, C, H, W]
#             video_data0 = video_data0.squeeze(
#                 0
#             )
#             video_data10 = video_data10.squeeze(
#                 0
#             )

#             logger.info(
#                 "text: {}".format(
#                     text
#                 )
#             )
#             logger.info(
#                 "name: {}".format(
#                     file_name
#                 )
#             )
#             logger.info(
#                 "output fps: {}".format(
#                     fps
#                 )
#             )

#             caption = text

#             # ---------------------------------------------------------
#             # Validate video tensor
#             # ---------------------------------------------------------
#             if video_data0.ndim != 4:
#                 raise ValueError(
#                     "Expected video_data0 shape [T, C, H, W], "
#                     "but got {}".format(
#                         tuple(video_data0.shape)
#                     )
#                 )

#             if video_data10.ndim != 4:
#                 raise ValueError(
#                     "Expected video_data10 shape [T, C, H, W], "
#                     "but got {}".format(
#                         tuple(video_data10.shape)
#                     )
#                 )

#             (
#                 num_frames,
#                 channels0,
#                 h0,
#                 w0,
#             ) = video_data0.shape

#             (
#                 num_frames1,
#                 channels1,
#                 h1,
#                 w1,
#             ) = video_data10.shape

#             if num_frames != num_frames1:
#                 raise ValueError(
#                     "The two input videos have different frame counts: "
#                     "{} and {}".format(
#                         num_frames,
#                         num_frames1,
#                     )
#                 )

#             logger.info(
#                 "Original input 0 shape: {}".format(
#                     tuple(video_data0.shape)
#                 )
#             )
#             logger.info(
#                 "Original input 1 shape: {}".format(
#                     tuple(video_data10.shape)
#                 )
#             )

#             # ---------------------------------------------------------
#             # Resize input videos to 720p
#             # ---------------------------------------------------------
#             video_data = torch.nn.functional.interpolate(
#                 video_data0,
#                 size=(720, 1280),
#                 mode="bilinear",
#                 align_corners=False,
#             )

#             video_data1 = torch.nn.functional.interpolate(
#                 video_data10,
#                 size=(720, 1280),
#                 mode="bilinear",
#                 align_corners=False,
#             )

#             (
#                 resized_frames,
#                 resized_channels,
#                 height,
#                 width,
#             ) = video_data.shape

#             logger.info(
#                 "Model input shape: {}".format(
#                     tuple(video_data.shape)
#                 )
#             )
#             logger.info(
#                 "Model input resolution: {}".format(
#                     (height, width)
#                 )
#             )

#             # ---------------------------------------------------------
#             # Construct model input
#             # ---------------------------------------------------------
#             pre_data = {
#                 "video_data": video_data,
#                 "y": caption,
#                 "video_data1": video_data1,
#             }

#             total_noise_levels = 900

#             setup_seed(
#                 666
#             )

#             if not torch.cuda.is_available():
#                 raise RuntimeError(
#                     "CUDA is required because collate_fn currently "
#                     "uses device 'cuda:0'."
#                 )

#             data_tensor = collate_fn(
#                 pre_data,
#                 "cuda:0",
#             )

#             # ---------------------------------------------------------
#             # Run inference or profiling
#             # ---------------------------------------------------------
#             if (
#                 self.profile_flops
#                 and not self._has_profiled
#             ):
#                 output = self.profile_model_test(
#                     data_tensor=data_tensor,
#                     total_noise_levels=total_noise_levels,
#                     num_frames=num_frames,
#                     height=height,
#                     width=width,
#                 )

#                 self._has_profiled = True

#             else:
#                 with torch.no_grad():
#                     output = self.run_model_test(
#                         data_tensor=data_tensor,
#                         total_noise_levels=total_noise_levels,
#                     )

#             # profile_only 模式：
#             # 只统计第一个 batch，不继续保存视频
#             if (
#                 self.profile_only
#                 and self._has_profiled
#             ):
#                 logger.info(
#                     "FLOPs profiling completed. "
#                     "Exiting because --profile_only is enabled."
#                 )
#                 break

#             # ---------------------------------------------------------
#             # Convert output
#             # ---------------------------------------------------------
#             output = tensor2vid(
#                 output
#             )

#             # [T, H, W, C] -> [T, C, H, W]
#             output = output.permute(
#                 0,
#                 3,
#                 1,
#                 2,
#             )

#             # Resize back to original resolution
#             output = torch.nn.functional.interpolate(
#                 output,
#                 size=(h0, w0),
#                 mode="bilinear",
#                 align_corners=False,
#             )

#             # [T, C, H, W] -> [T, H, W, C]
#             output = output.permute(
#                 0,
#                 2,
#                 3,
#                 1,
#             )

#             # ---------------------------------------------------------
#             # Optional color correction
#             # ---------------------------------------------------------
#             # output = adain_color_fix(
#             #     output,
#             #     video_data0,
#             # )

#             # ---------------------------------------------------------
#             # Save output
#             # ---------------------------------------------------------
#             save_video(
#                 output,
#                 self.result_dir,
#                 file_name,
#                 fps=fps,
#             )

#             output_path = os.path.join(
#                 self.result_dir,
#                 file_name,
#             )

#             logger.info(
#                 "Saved result to: {}".format(
#                     output_path
#                 )
#             )


# def parse_args():
#     parser = ArgumentParser()

#     parser.add_argument(
#         "--input_path",
#         required=True,
#         type=str,
#         help="input video dataset path",
#     )

#     parser.add_argument(
#         "--save_dir",
#         type=str,
#         default="results",
#         help="save directory",
#     )

#     parser.add_argument(
#         "--model_path",
#         type=str,
#         default="./pretrained_weight/model.pt",
#         help="model checkpoint path",
#     )

#     parser.add_argument(
#         "--upscale",
#         type=int,
#         default=1,
#         help="up-scale factor; currently unused",
#     )

#     parser.add_argument(
#         "--max_chunk_len",
#         type=int,
#         default=30,
#         help="number of frames in each input chunk",
#     )

#     parser.add_argument(
#         "--cfg",
#         type=float,
#         default=7.5,
#         help="classifier-free guidance scale",
#     )

#     parser.add_argument(
#         "--solver_mode",
#         type=str,
#         default="fast",
#         choices=[
#             "fast",
#             "normal",
#         ],
#         help="solver mode",
#     )

#     parser.add_argument(
#         "--steps",
#         type=int,
#         default=1,
#         help=(
#             "number of sampling steps; "
#             "fast mode always uses one step"
#         ),
#     )

#     # -----------------------------------------------------------------
#     # FLOPs profiling arguments
#     # -----------------------------------------------------------------
#     parser.add_argument(
#         "--profile_flops",
#         action="store_true",
#         help=(
#             "profile recognized FLOPs for the first video chunk"
#         ),
#     )

#     parser.add_argument(
#         "--profile_detail",
#         action="store_true",
#         help=(
#             "print detailed torch.profiler operator table"
#         ),
#     )

#     parser.add_argument(
#         "--profile_trace",
#         type=str,
#         default="",
#         help=(
#             "optional Chrome Trace output path, "
#             "for example ./profile/star_trace.json"
#         ),
#     )

#     parser.add_argument(
#         "--profile_only",
#         action="store_true",
#         help=(
#             "profile the first video chunk and exit "
#             "without saving the output video"
#         ),
#     )

#     return parser.parse_args()


# def main():
#     args = parse_args()

#     input_path = args.input_path
#     model_path = args.model_path
#     save_dir = args.save_dir
#     max_chunk_len = args.max_chunk_len

#     steps = args.steps
#     solver_mode = args.solver_mode
#     guide_scale = args.cfg

#     profile_flops = args.profile_flops
#     profile_detail = args.profile_detail
#     profile_trace = args.profile_trace
#     profile_only = args.profile_only

#     # profile_only 意味着必须开启 profiler
#     if profile_only:
#         profile_flops = True

#     test_dataset = TestPairedCaptionVideoDataset(
#         root_folders=[
#             input_path
#         ],
#         num_frames=max_chunk_len,
#     )

#     test_dataloader = torch.utils.data.DataLoader(
#         test_dataset,
#         num_workers=1,
#         batch_size=1,
#         shuffle=False,
#     )

#     star = STAR(
#         result_dir=save_dir,
#         model_path=model_path,
#         solver_mode=solver_mode,
#         steps=steps,
#         guide_scale=guide_scale,
#         max_chunk_len=max_chunk_len,
#         profile_flops=profile_flops,
#         profile_detail=profile_detail,
#         profile_trace=profile_trace,
#         profile_only=profile_only,
#     )

#     star.enhance_a_video(
#         test_dataloader
#     )


# if __name__ == "__main__":
#     main()