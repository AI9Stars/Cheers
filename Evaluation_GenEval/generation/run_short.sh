# replace the model ckpt

python ./generation/umm_generate.py \
--metadata_file prompts/evaluation_metadata.jsonl \
--model_path Cheers-CKPT/v1 \ 
--outdir "outputs/short" \
--cfg 9.5 \
--steps 80 \
--batch_size 4 

