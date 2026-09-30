### CogVideoX-based Model Inference
#### Step 1: Install the requirements
```
conda create -n star_cog python=3.10
conda activate star_cog
cd cogvideox-based/sat
pip install -r requirements.txt
```

#### Step 2: Download the pretrained model.
Download STAR from [HuggingFace](https://huggingface.co/SherryX/STAR).

Download VAE and T5 Encoder following this [instruction](https://github.com/THUDM/CogVideo/blob/main/sat/README_zh.md#cogvideox15-%E6%A8%A1%E5%9E%8B).


#### Step 3: Prepare testing data
You can put the testing videos in the `input/video/`.

As for the prompt, there are three options: 1. No prompt. 2. Automatically generate a prompt [using Pllava](https://github.com/hpcaitech/Open-Sora/tree/main/tools/caption#pllava-captioning). 3. Manually write the prompt. You can put the txt file in the `input/text/`.


#### Step 4: Change the cogfigs
You need to update the paths in `cogvideox-based/sat/configs/cogvideox_5b/cogvideox_5b_infer_sr.yaml` to match your local setup, including `load`, `output_dir`, `model_dir` of conditioner_config and `ckpt_path` of first_stage_config. Additionally, update the `test_dataset` path in sample_sr.py.


#### Step 5: Replace the transformer.py in sat packpage
Replace the `/cogvideo/lib/python3.9/site-packages/sat/model/transformer.py` in your enviroment with our provided [transformer.py](https://github.com/NJU-PCALab/STAR/blob/main/cogvideox-based/transformer.py).


#### Step 6: Running inference command
```
bash inference_sr.sh
```
