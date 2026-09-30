#! /bin/bash

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sample_sr.py --base configs/cogvideox_5b/cogvideox_5b_infer_sr.yaml"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"