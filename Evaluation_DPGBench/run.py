import json
import os
import subprocess
from functools import partial
import argparse
import datetime
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from UMM import build_model
from benchmarks import Dataset_eval
from tqdm import tqdm

# GET the number of GPUs on the node without importing libs like torch
def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if CUDA_VISIBLE_DEVICES != '':
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
        return gpu_list
    try:
        ps = subprocess.Popen(('nvidia-smi', '--list-gpus'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
        return list(range(int(output)))
    except:
        return []
 

RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK",1))

GPU_LIST = get_gpu_list()
if LOCAL_WORLD_SIZE > 1 and len(GPU_LIST):
    NGPU = len(GPU_LIST)
    assert NGPU >= LOCAL_WORLD_SIZE, "The number of processes should be less than or equal to the number of GPUs"
    GPU_PER_PROC = NGPU // LOCAL_WORLD_SIZE
    DEVICE_START_IDX = GPU_PER_PROC * LOCAL_RANK
    CUDA_VISIBLE_DEVICES = [str(i) for i in GPU_LIST[DEVICE_START_IDX: DEVICE_START_IDX + GPU_PER_PROC]]
    CUDA_VISIBLE_DEVICES = ','.join(CUDA_VISIBLE_DEVICES)
    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    print(
        f'RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE},'
        f'LOCAL_WORLD_SIZE: {LOCAL_WORLD_SIZE}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}'
    )



def parse_args():
    parser = argparse.ArgumentParser(description="input model and benchmarks", formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', help='Names of Geneval Benchmark Datasets(Geneval and ...)')
    parser.add_argument('--model', type=str, nargs='+', help='Names of UMM Models')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./Evaluation_DPGBench/outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer', 'eval'])
    parser.add_argument('--is-cot', type=bool, default=False)
    parser.add_argument('--is-sft', type=bool, default=False)
    parser.add_argument('--batchsize', type=int, default=4)  #####################
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--cfg', type=float, default=None)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--notion', type=str, default=None)

    # batch_size


    args = parser.parse_args()
    return args 


def main():
    args = parse_args()
    use_config, cfg = False, None

    if WORLD_SIZE > 1:
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 360000)))
        )

    for _, model_name in enumerate(args.model):
        eval_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        eval_id = f"{eval_id}_{args.alpha}_{args.cfg}_{args.steps}_{args.notion}_a-c-s-n"

        # pred_root = os.path.join(args.work_dir, model_name, eval_id)
        pred_root_meta = os.path.join(args.work_dir, model_name)
        os.makedirs(pred_root_meta, exist_ok=True)

        model = build_model(model_name, alpha=args.alpha, cfg=args.cfg, steps=args.steps, model_path=args.model_path)

        dataset_work_dir_list = []

        print(args.data)
        org_batch_size = args.batchsize
        for _, dataset_name in enumerate(args.data):
            
            dataset_work_dir = os.path.join(pred_root_meta, dataset_name, eval_id)
            if not os.path.exists(dataset_work_dir):
                os.makedirs(dataset_work_dir, exist_ok=True)

            dataset_meta = Dataset_eval[dataset_name]()
            if dataset_meta.gen_num_images() < args.batchsize:
                args.batchsize = dataset_meta.gen_num_images()

            try:
                if WORLD_SIZE > 1:
                    dist.barrier()

                if args.mode != "eval":

                    if dataset_meta.datatype() == "gen":
                        total_metadatas = len(dataset_meta)
                        prompts_per_gpu = (total_metadatas + WORLD_SIZE - 1) // WORLD_SIZE
                        start = RANK * prompts_per_gpu
                        end = min(start + prompts_per_gpu, total_metadatas)
                        print(f"GPU {RANK}: Processing {end - start} prompts (indices {start} to {end - 1}), total_metadatas: {total_metadatas}")
                        for idx in tqdm(range(start, end), desc=f"RANK: {RANK}, dataset: {dataset_name}, start:{start}, end: {end}, all: {total_metadatas}"):
                            prompt = dataset_meta[idx]
                            image_list = []
                            print(f"GPU {RANK} processing prompt {idx - start + 1}/{end - start}: '{prompt}'")
                            for i in range(dataset_meta.gen_num_images() // args.batchsize):
                                tmp_image_list = model.gen_t2i(prompt, args.batchsize, dataset_name)
                                image_list.extend(tmp_image_list)
                            # breakpoint()
                            dataset_meta.output_form(idx, dataset_work_dir, image_list)
                    elif dataset_meta.datatype() == "edit":
                        total_metadatas = len(dataset_meta)
                        prompts_per_gpu = (total_metadatas + WORLD_SIZE - 1) // WORLD_SIZE
                        start = RANK * prompts_per_gpu
                        end = min(start + prompts_per_gpu, total_metadatas)
                        print(f"GPU {RANK}: Processing {end - start} prompts (indices {start} to {end - 1})")
                        for idx in tqdm(range(start, end), desc=f"RANK: {RANK}, dataset: {dataset_name}, start:{start}, end: {end}, all: {total_metadatas}"):
                            image_list = []
                            concat_image_list = []
                            prompt, input_images = dataset_meta[idx]
                            print(f"GPU {RANK} processing prompt {idx - start + 1}/{end - start}: '{prompt}'")
                            for i in range(dataset_meta.gen_num_images() // args.batchsize):
                                # breakpoint()
                                tmp_image_list, tmp_concat_image_list = model.gen_i2i(prompt, input_images, args.batchsize, dataset_name)
                                image_list.extend(tmp_image_list)
                                concat_image_list.extend(tmp_concat_image_list)
                            dataset_meta.output_form(idx, dataset_work_dir, image_list)
                            dataset_meta.output_form_concat(idx, dataset_work_dir, concat_image_list)




                    print(f"GPU {RANK} has completed all tasks")
                    dist.barrier()

                if WORLD_SIZE > 1:
                    dist.barrier()

                # if RANK == 0:
                #     # dataset_meta.eval_result(dataset_work_dir)
                #     pass

                dataset_work_dir_list.append(dataset_work_dir)


            except Exception as e:
                print("error: ", e)
                continue

            args.batchsize = org_batch_size

    if WORLD_SIZE > 1:
        dist.destroy_process_group()

    return dataset_work_dir_list


if __name__ == '__main__':
    main()
