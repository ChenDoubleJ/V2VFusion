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

from inference_utils import *

logger = get_logger()


class STAR():
    def __init__(self, 
                 result_dir='./results/',
                 file_name='000_video.mp4',
                 model_path='',
                 solver_mode='fast',
                 steps=1,
                 guide_scale=7.5,
                 max_chunk_len=32,
                 ):
        self.model_path=model_path
        logger.info('checkpoint_path: {}'.format(self.model_path))

        self.result_dir = result_dir
        self.file_name = file_name
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

    def enhance_a_video(self, video_path, prompt, video_path1):
        logger.info('input video path: {}'.format(video_path))
        logger.info('input video1 path: {}'.format(video_path1))
        text = prompt
        # text1 = prompt1
        logger.info('text: {}'.format(text))
        # logger.info('text1: {}'.format(text1))
        # logger.info('positive_prompt: {}'.format(self.model.positive_prompt))
        # caption = text + self.model.positive_prompt
        # caption1 = text1 + self.model.positive_prompt
        caption = text
        # caption1 = text1

        input_frames, input_fps = load_video(video_path)
        input_frames1, input_fps1 = load_video(video_path1)
        logger.info('input fps: {}'.format(input_fps))
        logger.info('input fps1: {}'.format(input_fps1))

        video_data0 = preprocess(input_frames)
        video_data10 = preprocess(input_frames1)
# ***
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
        save_video(output, self.result_dir, self.file_name, fps=input_fps)
        return os.path.join(self.result_dir, self.file_name)
    

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--input_path", required=True, type=str, help="input video path")
    parser.add_argument("--input_path1", required=True, type=str, help="input video path")
    parser.add_argument("--save_dir", type=str, default='results', help="save directory")
    parser.add_argument("--file_name", type=str, help="file name")
    parser.add_argument("--model_path", type=str, default='./pretrained_weight/model.pt', help="model path")
    parser.add_argument("--prompt", type=str, default='a good video', help="prompt")
    # parser.add_argument("--prompt1", type=str, default='a good video', help="prompt")
    parser.add_argument("--upscale", type=int, default=1, help='up-scale')
    parser.add_argument("--max_chunk_len", type=int, default=32, help='max_chunk_len')

    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--solver_mode", type=str, default='fast', help='fast | normal')
    parser.add_argument("--steps", type=int, default=1)

    return parser.parse_args()

def main():
    
    args = parse_args()

    input_path = args.input_path
    prompt = args.prompt
    input_path1 = args.input_path1
    model_path = args.model_path
    save_dir = args.save_dir
    file_name = args.file_name
    upscale = args.upscale
    max_chunk_len = args.max_chunk_len

    steps = args.steps
    solver_mode = args.solver_mode
    guide_scale = args.cfg

    assert solver_mode in ('fast', 'normal')

    star = STAR(
                result_dir=save_dir,
                file_name=file_name,
                model_path=model_path,
                solver_mode=solver_mode,
                steps=steps,
                guide_scale=guide_scale,
                max_chunk_len=max_chunk_len,
                )

    star.enhance_a_video(input_path, prompt, input_path1)

if __name__ == '__main__':
    main()
