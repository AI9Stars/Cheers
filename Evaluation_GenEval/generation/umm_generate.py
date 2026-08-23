import os
import json
import argparse
import torch

from tqdm import tqdm
from torchvision.utils import save_image
from transformers import AutoModelForCausalLM, AutoProcessor

import torch.multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="JSONL file containing lines of metadata for each prompt")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True,
                        help="Pretrained model ckpt path")
    parser.add_argument("--batch_size", type=int, required=True)

    # 可选：限制使用的 GPU 数量（默认用全部可见 GPU）
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all visible GPUs)")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG Value (default: 4)")
    parser.add_argument("--steps", type=int, default=50, help="Steps Value (default: 50)")
    return parser.parse_args()


def load_metadatas(metadata_file: str):
    # 建议：JSONL 用逐行读取（避免一次性 json.load）
    metadatas = []
    with open(metadata_file, "r", encoding="utf-8") as fp:
        for line_idx, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue
            metadatas.append((line_idx, json.loads(line)))
    return metadatas


def worker(rank: int, world_size: int, opt, metadatas):
    """
    rank: 进程编号 [0..world_size-1]
    world_size: GPU/进程数
    """
    # 降低多进程 CPU 线程争抢
    torch.set_num_threads(1)

    # 绑定 GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        # 没 GPU 就退化单进程 CPU（此时 world_size 应该为 1）
        device = torch.device("cpu")

    # 每个进程各自加载一次模型（每 GPU 一个进程的标准做法）
    processor = AutoProcessor.from_pretrained(opt.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(opt.model_path, trust_remote_code=True)

    if device.type == "cuda":
        model = model.to(device).to(torch.bfloat16)
    else:
        model = model.to(device)

    model.eval()

    # 分片：按 index % world_size == rank
    local_items = [(idx, md) for (idx, md) in metadatas if (idx % world_size) == rank]

    # tqdm 只在各自进程显示自己的进度（position 防止互相覆盖）
    pbar = tqdm(local_items, desc=f"GPU{rank}", position=rank, leave=True)

    for index, metadata in pbar:
        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata["prompt"]
        batch_size = opt.batch_size

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        # 写入 metadata（这里原脚本叫 metadata.jsonl 但写的是 json；我保持你的行为不改动）
        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False)

        messages_batch = [[{"role": "user", "content": prompt}] for _ in range(batch_size)]
        images_batch = [None for _ in range(batch_size)]

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]

        inputs = processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            add_im_start_id=True,
        )

        # move tensors to device
        inputs = {k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}

        gen_config = {
            "max_length": 500,
            "cfg_scale": opt.cfg,
            "temperature": 0.0,
            "num_inference_steps": opt.steps,
        }
        
        print(f'gen_config {gen_config}')
        inputs.update(gen_config)

        with torch.no_grad():
            generated = model.generate(**inputs)

        images = generated["images"][0]
        for image_num, current_img in enumerate(images):
            current_img = current_img.clamp(0.0, 1.0)
            save_image(current_img, os.path.join(sample_path, f"{image_num:05}.png"))

    # 可选：释放显存
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main(opt):
    os.makedirs(opt.outdir, exist_ok=True)

    metadatas = load_metadatas(opt.metadata_file)

    # 决定 GPU 数量
    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        world_size = visible if opt.num_gpus is None else min(opt.num_gpus, visible)
        if world_size < 1:
            raise RuntimeError("No CUDA devices found.")
        print(f"\n✅ Using {world_size} GPU process(es).")
    else:
        world_size = 1
        print("\n⚠️ CUDA not available. Falling back to 1 CPU process.")

    # Windows 上建议使用 spawn；Linux 默认 fork 但用 spawn 更稳（尤其 transformers + cuda）
    mp.set_start_method("spawn", force=True)

    if world_size == 1:
        worker(0, 1, opt, metadatas)
    else:
        mp.spawn(
            worker,
            args=(world_size, opt, metadatas),
            nprocs=world_size,
            join=True,
        )

    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)