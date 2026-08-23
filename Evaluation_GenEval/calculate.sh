
IMAGE_FOLDER="output_image_path"
echo $MODEL_NAME
python evaluation/evaluate_images.py \
    $IMAGE_FOLDER \
    --outfile "results/result.jsonl" \
    --model-path "models/"