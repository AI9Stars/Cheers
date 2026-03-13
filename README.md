<div align="center">

# ***Cheers <img src="fig/logo.png" width="25"> : Decoupling Patch Details from Semantic Representations Enables Unified Multimodal Comprehension and Generation*** 
<p align="center">
        🤗 <a href="https://huggingface.co/ai9stars/Cheers">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="1">Paper</a>&nbsp&nbsp
</p>

Yichen Zhang<sup>1*</sup>, [Da Peng](https://pengda02.github.io/)<sup>2*</sup>, [Zonghao Guo](https://scholar.google.com/citations?user=h1I6LJcAAAAJ&hl=zh-CN)<sup>1†</sup>, Zijian Zhang<sup>3</sup>, Xuesong Yang<sup>3</sup>,

Tong Sun<sup>3</sup>, Shichu Sun<sup>3</sup>, Yidan Zhang<sup>3</sup>, Yanghao Li<sup>1</sup>, Haiyan Zhao<sup>1</sup>, Wang Xu<sup>1</sup>,

Qi Shi<sup>1</sup>, Yangang Sun<sup>1</sup>, Chi Chen<sup>1</sup>, Shuo Wang<sup>1</sup>, Yukun Yan<sup>1</sup>, Xu Han<sup>1</sup>,

Qiang Ma<sup>1</sup>, [Wei Ke](https://scholar.google.com/citations?hl=en&user=BENt-uEAAAAJ)<sup>2</sup>, Liang Wang<sup>3</sup>, Zhiyuan Liu<sup>1</sup>, Maosong Sun<sup>1</sup>

<sup>1</sup>Tsinghua University, 
<sup>2</sup>Xi'an Jiaotong University, 
<sup>3</sup>University of Chinese Academy of Sciences

\* Equal contribution
† Corresponding author

</div>
<img src="fig/case.png" width="100%">


## 🌟 What is ***Cheers***?
A recent cutting-edge topic in multimodal modeling is to unify visual comprehension and generation within a single model. However, the two tasks demand mismatched decoding regimes and visual representations, making it non-trivial to jointly optimize within a shared feature space. In this work, we present ***Cheers***, a unified multimodal model that decouples patch-level details from semantic representations, thereby stabilizing semantics for multimodal understanding and improving fidelity for image generation via gated detail residuals. ***Cheers*** includes three key components: (i) a unified vision tokenizer that encodes and compresses image latent states into semantic tokens for efficient LLM conditioning, (ii) an LLM-based Transformer that unifies autoregressive decoding for text generation and diffusion decoding for image generation, and (iii) a cascaded flow matching head that decodes visual semantics first and then injects semantically gated detail residuals from the vision tokenizer to refine high-frequency content. Experiments on popular benchmarks demonstrate that ***Cheers*** matches or surpasses advanced UMMs in both visual understanding and generation. Notably, ***Cheers*** outperforms the Tar-1.5B on the popular benchmarks GenEval and MMBench, while requiring only 20% of the training cost, indicating effective and efficient (i.e., 4x token compression) unified multimodal modeling.

## Model Architecture
<img src="fig/model.png" width="100%">

## 🔥 News

-[2026/03/16] 📢 The ***Cheers*** paper is officially released.

-[2026/03/16] 🛠 We open-source the training pipeline and evaluation code.

-[2026/03/16] 📦 The model checkpoints of ***Cheers*** are now available.

## 🚀 Quick Start
### Set up a new virtual environment
```bash
conda create -n cheers python=3.10 -y
conda activate cheers
pip install -r requirements.txt
```

### Inference
```bash
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
ckpt = "Your Local Checkpoints Path"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
model.to(device)
model = model.to(torch.bfloat16)
model.eval()
```
1️⃣ Text-to-image generation 
```bash
content = "Your instruction."
images_batch = [None]
```
2️⃣ Image Understanding
```bash
content = "<im_start><image><im_end>\n Your instruction."
img = Image.open("image_path")
images_batch = [img,]
```
3️⃣ Text-only Question Answering
```bash
content = "Your instruction."
images_batch = [None]
```
Then run the following code:
```bash
gen_config = {
    "max_length": 300,
    "cfg_scale": 9.5, # if generation
    "temperature": 0.0,
    "num_inference_steps": 80, # if use# if generation
    "alpha": 0.5, # if generation
    "edit_image": False # if generation
    }

inputs.update(gen_config)
generated = model.generate(**inputs)
input_ids = generated["input_ids"]

images = generated["images"][0] # if generation
current_img = images[0] # if generation
current_img = current_img.clamp(0.0, 1.0) # if generation
save_image(current_img, f"outputs/case_.png") # if generation
print(processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)) # if understanding or text-only qa
```
Alternatively, you can directly run the code in [`Inference/`](./Inference) for a quick demo.

## Training
Please follow the [VeOmni](https://github.com/ByteDance-Seed/VeOmni) framework guidelines to set up the training environment. The training workspace is located in the [`Training/`](./Training) directory. Then you can run the following scripts:
```bash
bash train_align.sh # for alignment
```
or
```bash
bash train_sft.sh # for training all parameters except the VAE.
```
Notably, the training data format can follow the template at [`Training/data/format.jsonl`](Training/data/format.jsonl). Please also remember to update the training configuration in [`Training/configs/multimodal/cheers/und_gen_train/`](Training/configs/multimodal/cheers/und_gen_train/).

## ***Evaluation***

### **Understanding**
Please follow the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) framework guidelines to set up the evaluation environment. The evaluation workspace is located in the [`Evaluation_Understanding/`](./Evaluation_Understanding) directory. Then you can run the following scripts:

```bash
torchrun --master_port=19507 --nproc-per-node=8  run.py --data \
    MathVista_MINI MMBench_DEV_EN_V11 SEEDBench_IMG \
    MMStar POPE RealWorldQA MMMU_DEV_VAL ScienceQA_TEST  \
    AI2D_TEST OCRBench TextVQA_VAL ChartQA_TEST \
    --model Cheers --verbose 
```
Similarly, you can directly run the following script to perform the evaluation:
```bash
bash eval.sh
```
Please make sure to update the dataset path in [`eval.sh`](./Evaluation_Understanding/eval.sh) and the model path in [`vlmeval/config.py`](./Evaluation_Understanding/vlmeval/config.py) before running the script.

### **GenEval**
Please follow the [GenEval](https://github.com/djghosh13/geneval) framework guidelines to set up the GenEval evaluation environment. The evaluation workspace is located in the [`Evaluation_GenEval/`](./Evaluation_GenEval) directory. Before running the evaluation, please download the Mask2Former object detector and place it in [`models`](./Evaluation_GenEval/models). Then you can run the following scripts:
```bash
bash generation/run_short.sh
```
or
```bash
bash generation/run_long.sh # for rewrite prompt
```
Use the follow script to get the final score:
```bash
bash calculate.sh
```
### **DPGBench**
Please follow the [ELLA](https://github.com/TencentQQGYLab/ELLA) framework guidelines to set up the DPGBench evaluation environment. Before running the evaluation, please download the mplug and place it in [`benchmarks/dpg/mplug_visual-question-answering_coco_large_en`](./Evaluation_DPGBench/benchmarks/dpg/mplug_visual-question-answering_coco_large_en). Then you can run the following scripts:
```bash
bash Evaluation_DPGBench/scripts/dpg_gen.sh
```
Then:
```bash
bash Evaluation_DPGBench/scripts/dpg_eval.sh # Remember to replace the image folder
```
## 🧩 To-Do List
- [x] Release the **Inference Scripts** and **Checkpoints**
- [x] Release the **Training Scripts** using the VeOmni framework
- [x] Release the **Evaluation Scripts**
- [ ] Release the **Training Data** 
- [ ] Release **Cheers v1.1** — maintaining strong understanding performance while further improving generation quality

---
This repo benefits from [VeOmni](https://github.com/ByteDance-Seed/VeOmni), [VLMEvalKit
](https://github.com/open-compass/VLMEvalKit), [GenEval](https://github.com/djghosh13/geneval) and [ELLA](https://github.com/TencentQQGYLab/ELLA). Thanks for their wonderful works.

## 📬 Contact
For any questions or collaborations, feel free to contact us : )

<p align="left">
📧 <a href="MetaPDa@gmail.com">MetaPDa@gmail.com</a>&nbsp&nbsp | &nbsp&nbsp 📧 <a href="guozonghao96@outlook.com">guozonghao96@outlook.com</a>&nbsp&nbsp | &nbsp&nbsp 📧 <a href="yichen0zhang@gmail.com">yichen0zhang@gmail.com</a>&nbsp&nbsp
</p>
