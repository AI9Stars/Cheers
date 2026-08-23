# replace the model ckpt

python ./generation/umm_generate.py \
--metadata_file prompts/evaluation_metadata_long.jsonl \
--model_path Cheers-CKPT/v1 \
--outdir "outputs/long" \
--cfg 9.5 \
--steps 80 \
--batch_size 4 
