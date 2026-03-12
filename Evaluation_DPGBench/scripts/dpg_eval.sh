RESOLUTION=512
PIC_NUM=${PIC_NUM:-4}
PROCESSES=${PROCESSES:-8}
PORT=${PORT:-29621}


IMAGE_ROOT_PATH="your_image_dir"
accelerate launch --num_machines 1 --num_processes $PROCESSES --mixed_precision "fp16" --main_process_port $PORT \
  ./Evaluation_DPGBench/benchmarks/dpg/compute_dpg_bench.py \
  --image-root-path $IMAGE_ROOT_PATH \
  --resolution $RESOLUTION \
  --pic-num $PIC_NUM \
  --vqa-model mplug