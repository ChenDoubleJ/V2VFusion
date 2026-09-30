CUDA_VISIBLE_DEVICES=2 python \
    ./video_fusion/scripts/inference.py \
    --solver_mode 'fast' \
    --steps 5 \
    --input_path ./dataset/dataset-github/visible-infrared \
    --model_path ./out-use/checkpoint-15000/model.safetensors \
    --max_chunk_len 30 \
    --save_dir ./results-github \
    --cfg 1.0

# for dir in ./dataset/dataset-github/; do
#     name=$(basename "$dir")
#     echo "Processing $name"

#     CUDA_VISIBLE_DEVICES=2 python ./video_fusion/scripts/inference.py \
#         --solver_mode 'fast' \
#         --steps 5 \
#         --input_path "$dir" \
#         --model_path ./out-use/checkpoint-12000/model.safetensors \
#         --max_chunk_len 30 \
#         --save_dir "./results-github/$name" \
#         --cfg 1.0 
# done