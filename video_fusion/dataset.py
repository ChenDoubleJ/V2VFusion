import os
import random
import glob
import torchvision
from einops import rearrange
from torch.utils import data as data
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class PairedCaptionVideoDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            null_text_ratio=0.5,
            num_frames=16
    ):
        super(PairedCaptionVideoDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.lr_list1 = []
        self.gt_list = []
        self.tag_path_list = []
        self.num_frames = num_frames

        # root_folders = root_folders.split(',')
        for root_folder in root_folders:
            lr_path = root_folder +'/lq'
            lr_path1 = root_folder +'/lq1'
            tag_path = root_folder +'/text'
            gt_path = root_folder +'/gt'

            self.lr_list += glob.glob(os.path.join(lr_path, '*.mp4'))
            self.lr_list1 += glob.glob(os.path.join(lr_path1, '*.mp4'))
            self.gt_list += glob.glob(os.path.join(gt_path, '*.mp4'))
            self.tag_path_list += glob.glob(os.path.join(tag_path, '*.txt'))

        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)
        assert len(self.lr_list) == len(self.lr_list1)
    # def __getitem__(self, index):

    #     gt_path = self.gt_list[index]
    #     vframes_gt, _, info = torchvision.io.read_video(filename=gt_path, pts_unit="sec", output_format="TCHW")
    #     fps = info['video_fps']
    #     vframes_gt = (rearrange(vframes_gt, "T C H W -> C T H W") / 255) * 2 - 1

    #     lq_path = self.lr_list[index]
    #     vframes_lq, _, _ = torchvision.io.read_video(filename=lq_path, pts_unit="sec", output_format="TCHW")
    #     vframes_lq = (rearrange(vframes_lq, "T C H W -> C T H W") / 255) * 2 - 1

    #     if random.random() < self.null_text_ratio:
    #         tag = ''
    #     else:
    #         tag_path = self.tag_path_list[index]
    #         with open(tag_path, 'r', encoding='utf-8') as file:
    #             tag = file.read()

    #     return {"gt": vframes_gt[:, :self.num_frames, :, :], "lq": vframes_lq[:, :self.num_frames, :, :], "text": tag, 'fps': fps}

    # def __len__(self):
    #     return len(self.gt_list)

    def __getitem__(self, index):

        gt_path = self.gt_list[index]
        vframes_gt, _, info = torchvision.io.read_video(
            filename=gt_path, pts_unit="sec", output_format="TCHW"
        )
        fps = info['video_fps']

        vframes_gt = (rearrange(vframes_gt, "T C H W -> C T H W") / 255) * 2 - 1

        # resize GT -> 512x512
        vframes_gt = rearrange(vframes_gt, "C T H W -> T C H W")
        vframes_gt = F.interpolate(vframes_gt, size=(720, 1280), mode="bilinear", align_corners=False)
        vframes_gt = rearrange(vframes_gt, "T C H W -> C T H W")

        lq_path = self.lr_list[index]
        vframes_lq, _, _ = torchvision.io.read_video(
            filename=lq_path, pts_unit="sec", output_format="TCHW"
        )
        vframes_lq = (rearrange(vframes_lq, "T C H W -> C T H W") / 255) * 2 - 1

        # resize LQ -> 512x512
        vframes_lq = rearrange(vframes_lq, "C T H W -> T C H W")
        vframes_lq = F.interpolate(vframes_lq, size=(720,1280), mode="bilinear", align_corners=False)
        vframes_lq = rearrange(vframes_lq, "T C H W -> C T H W")


        lq_path1 = self.lr_list1[index]
        vframes_lq1, _, _ = torchvision.io.read_video(
            filename=lq_path1, pts_unit="sec", output_format="TCHW"
        )
        vframes_lq1 = (rearrange(vframes_lq1, "T C H W -> C T H W") / 255) * 2 - 1

        # resize LQ1 -> 512x512
        vframes_lq1 = rearrange(vframes_lq1, "C T H W -> T C H W")
        vframes_lq1 = F.interpolate(vframes_lq1, size=(720,1280), mode="bilinear", align_corners=False)
        vframes_lq1 = rearrange(vframes_lq1, "T C H W -> C T H W")


        if random.random() < self.null_text_ratio:
            tag = ''
        else:
            tag_path = self.tag_path_list[index]
            with open(tag_path, 'r', encoding='utf-8') as file:
                tag = file.read()

        return {
            "gt": vframes_gt[:, :self.num_frames, :, :],
            "lq": vframes_lq[:, :self.num_frames, :, :],
            "lq1": vframes_lq1[:, :self.num_frames, :, :],
            "text": tag,
            "fps": fps
        }
    def __len__(self):
        return len(self.gt_list)
class TestPairedCaptionVideoDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            null_text_ratio=0.5,
            num_frames=16
    ):
        super(TestPairedCaptionVideoDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.lr_list1 = []
        # self.gt_list = []
        self.tag_path_list = []
        self.num_frames = num_frames

        # root_folders = root_folders.split(',')
        for root_folder in root_folders:
            lr_path = root_folder +'/lq'
            lr_path1 = root_folder +'/lq1'
            tag_path = root_folder +'/text'

            self.lr_list += glob.glob(os.path.join(lr_path, '*.mp4'))
            self.lr_list1 += glob.glob(os.path.join(lr_path1, '*.mp4'))
            self.tag_path_list += glob.glob(os.path.join(tag_path, '*.txt'))

        # assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)
        assert len(self.lr_list) == len(self.lr_list1)
    def __getitem__(self, index):
        lq_path = self.lr_list[index]
        video_name = os.path.basename(lq_path)
        vframes_lq, _, info = torchvision.io.read_video(
            filename=lq_path, pts_unit="sec", output_format="TCHW"
        )
        fps = info['video_fps']
        vframes_lq = (rearrange(vframes_lq, "T C H W -> C T H W") / 255) * 2 - 1

        # resize LQ -> 512x512
        vframes_lq = rearrange(vframes_lq, "C T H W -> T C H W")
        # vframes_lq = F.interpolate(vframes_lq, size=(720,1280), mode="bilinear", align_corners=False)
        # vframes_lq = rearrange(vframes_lq, "T C H W -> C T H W")


        lq_path1 = self.lr_list1[index]
        vframes_lq1, _, _ = torchvision.io.read_video(
            filename=lq_path1, pts_unit="sec", output_format="TCHW"
        )
        vframes_lq1 = (rearrange(vframes_lq1, "T C H W -> C T H W") / 255) * 2 - 1

        # resize LQ1 -> 512x512
        vframes_lq1 = rearrange(vframes_lq1, "C T H W -> T C H W")
        # vframes_lq1 = F.interpolate(vframes_lq1, size=(720,1280), mode="bilinear", align_corners=False)
        # vframes_lq1 = rearrange(vframes_lq1, "T C H W -> C T H W")


        # if random.random() < self.null_text_ratio:
        #     tag = ''
        # else:
        tag_path = self.tag_path_list[index]
        with open(tag_path, 'r', encoding='utf-8') as file:
            tag = file.read()

        return {
            "lq": vframes_lq[:, :self.num_frames, :, :],
            "lq1": vframes_lq1[:, :self.num_frames, :, :],
            "text": tag,
            "fps": fps,
            "name":video_name
        }
    def __len__(self):
        return len(self.lr_list)
class PairedCaptionImageDataset(data.Dataset):
    def __init__(
            self,
            root_folder=None,
    ):
        super(PairedCaptionImageDataset, self).__init__()

        self.lr_list = []
        self.gt_list = []
        self.tag_path_list = []

        lr_path = root_folder +'/sr_bicubic'
        gt_path = root_folder +'/gt'

        self.lr_list += glob.glob(os.path.join(lr_path, '*.png'))
        self.gt_list += glob.glob(os.path.join(gt_path, '*.png'))

        assert len(self.lr_list) == len(self.gt_list)

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])

        # Define the crop size (e.g., 256x256)
        crop_size = (720, 1280)

        # CenterCrop transform
        self.center_crop = transforms.CenterCrop(crop_size)

    def __getitem__(self, index):

        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.center_crop(self.img_preproc(gt_img))
        
        lq_path = self.lr_list[index]
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.center_crop(self.img_preproc(lq_img))

        example = dict()
        
        example["lq"] = (lq_img.squeeze(0) * 2.0 - 1.0).unsqueeze(1)
        example["gt"] = (gt_img.squeeze(0) * 2.0 - 1.0).unsqueeze(1)
        example["text"] = ""

        return example

    def __len__(self):
        return len(self.gt_list)