<div align="center">

# ***Cheers: Decoupling Patch Details from Semantic Representations Enables Unified Multimodal Comprehension and Generation*** 

Yichen Zhang<sup>1</sup>, Da Peng<sup>2</sup>, [Zonghao Guo](https://https://guozonghao96.github.io/)<sup>1</sup>, zijian zhang<sup>3</sup>,  Xuesong Yang<sup>3</sup>, 

Tong sun<sup>3</sup>, Shichu Sun<sup>3</sup>, Yidan Zhang<sup>3</sup>, Yanghao Li<sup></sup>, Haiyan Zhao<sup></sup>, Wang Xu<sup></sup>, 

Qi Shi<sup>1</sup>, Yangang Sun<sup>1</sup>, Chi Chen<sup>1</sup>, Shuo Wang<sup>1</sup>, Yukun Yan<sup></sup>, Xu Han<sup>1</sup>, 

Qiang Ma<sup>1</sup>, Wei ke<sup>2</sup>, Liang Wang<sup>3</sup>, Zhiyuan Liu<sup>1</sup>, Maosong Sun<sup>1</sup>, 

<sup>1</sup> Tsinghua University
<sup>2</sup> Xi'an Jiaotong University, 
<sup>3</sup> University of Chinese Academy of Sciences, 

</div>


## 🌟 What is ***Cheers***?
A recent cutting-edge topic in multimodal modeling is to unify visual comprehension and generation within a single model. However, the two tasks demand mismatched decoding regimes and visual representations, making it non-trivial to jointly optimize within a shared feature space. In this work, we present ***Cheers***, a unified multimodal model that decouples patch-level details from semantic representations, thereby stabilizing semantics for multimodal understanding and improving fidelity for image generation via gated detail residuals. ***Cheers*** includes three key components: (i) a unified vision tokenizer that encodes and compresses image latent states into semantic tokens for efficient LLM conditioning, (ii) an LLM-based Transformer that unifies autoregressive decoding for text generation and diffusion decoding for image generation, and (iii) a cascaded flow matching head that decodes visual semantics first and then injects semantically gated detail residuals from the vision tokenizer to refine high-frequency content. Experiments on popular benchmarks demonstrate that ***Cheers*** matches or surpasses advanced UMMs in both visual understanding and generation. Notably, ***Cheers*** outperforms the Tar-1.5B on the popular benchmarks GenEval and MMBench, while requiring only 20% of the training cost, indicating effective and efficient (i.e., 4x token compression) unified multimodal modeling. We will release all code and data for future research. 


## 🔥 News


## 🚀 Quick Start
### Set up a new virtual environment
```bash
conda create -n cheers python=3.10 -y
conda activate cheers
```



## 🧩 To-Do List

- [ ] Release the **Inference Scripts** and **Checkpoints**
- [ ] Release the **Training Scripts** using the VeOmni framework
- [ ] Release the **Training Data Recipe** 

---

## 📬 Contact
For any questions or collaborations, feel free to contact me : )

📧 **[MetaPDa@gmail.com](MetaPDa@gmail.com)**