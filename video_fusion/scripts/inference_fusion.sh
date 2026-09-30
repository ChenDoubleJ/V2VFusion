video_folder_path='./input/videoA'
video_folder_path1='./input/videoB'

txt_file_path='./input/text/prompt.txt'
# txt_file_path1='./input/textB/prompt.txt'

# Get all .mp4 files in videoA
mapfile -t mp4_files < <(find "$video_folder_path" -type f -name "*.mp4" | sort)

# Print video list
echo "Video files to be processed:"
for mp4_file in "${mp4_files[@]}"; do
    echo "$mp4_file"
done

# Read prompt files
mapfile -t lines < <(grep -v '^\s*$' "$txt_file_path")

# Frame length
frame_length=30

echo "Number of videos: ${#mp4_files[@]}"
echo "Prompt lines: ${#lines[@]}"

# Check match
if [ ${#mp4_files[@]} -ne ${#lines[@]} ]; then
    echo "Mismatch between video files and prompt lines."
    exit 1
fi

# Loop
for i in "${!mp4_files[@]}"; do

    mp4_file="${mp4_files[$i]}"
    line="${lines[$i]}"

    # filename
    file_name=$(basename "$mp4_file")

    # second video path
    mp4_file1="${video_folder_path1}/${file_name}"

    # remove extension
    file_base=$(basename "$mp4_file" .mp4)

    echo "Processing:"
    echo "VideoA: $mp4_file"
    echo "VideoB: $mp4_file1"
    echo "Prompt: $line"
 # --model_path ./pretrained_weight/model.pt \
    python \
        ./video_fusion/scripts/inference_fusion.py \
        --solver_mode 'fast' \
        --steps 20 \
        --input_path "${mp4_file}" \
        --input_path1 "${mp4_file1}" \
        --model_path ./out0/checkpoint-7000/model.safetensors \
        --prompt "${line}" \
        --max_chunk_len ${frame_length} \
        --file_name "${file_base}.mp4" \
        --save_dir ./results \
        --cfg 1.0

done

echo "All videos processed successfully."