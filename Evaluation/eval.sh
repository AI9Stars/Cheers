
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_TRACE_BUFFER_SIZE=4194304
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_IDS[@]}

export LMUData=/local/to/LMUData
echo $LMUData

MODEL_NAME=Cheers
MASTER_PORT=19507

ACCELERATE_CPU_AFFINITY=1 torchrun --master_port=$MASTER_PORT --nproc-per-node=$NUM_GPUS run.py --data \
    MathVista_MINI MMBench_DEV_EN_V11 SEEDBench_IMG \
    MMStar POPE RealWorldQA MMMU_DEV_VAL ScienceQA_TEST  \
    AI2D_TEST OCRBench TextVQA_VAL ChartQA_TEST \
    --model $MODEL_NAME --verbose 