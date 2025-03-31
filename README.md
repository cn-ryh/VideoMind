<p align="center">
  <img width="350" src=".github/logo.png">
</p>

<h2 align="center">VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2503.13444" target="_blank"><img src="https://img.shields.io/badge/arXiv-2503.13444-red"></a>
  <a href="https://videomind.github.io/" target="_blank"><img src="https://img.shields.io/badge/Project-Page-brightgreen"></a>
  <a href="https://huggingface.co/collections/yeliudev/videomind-67dd41f42c57f0e7433afb36" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
  <a href="https://huggingface.co/datasets/yeliudev/VideoMind-Dataset" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a>
  <a href="https://huggingface.co/spaces/yeliudev/VideoMind-2B" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg"></a>
</p>

<p align="center">
  <a href="https://yeliu.dev/" target="_blank">Ye Liu</a><sup>1&dagger;</sup>, <a href="https://qhlin.me/" target="_blank">Kevin Qinghong Lin</a><sup>2&dagger;</sup>, <a href="https://web.comp.polyu.edu.hk/chencw/" target="_blank">Chang Wen Chen</a><sup>1</sup>, <a href="https://sites.google.com/view/showlab" target="_blank">Mike Zheng Shou</a><sup>2</sup>
  <p align="center"><sup>1</sup>The Hong Kong Polytechnic University <sup>2</sup>Show Lab, National University of Singapore</p>
</p>

**VideoMind** is a multi-modal agent framework that enhances video reasoning by emulating *human-like* processes, such as *breaking down tasks*, *localizing* and *verifying* moments, and *synthesizing answers*. This approach addresses the unique challenges of temporal-grounded reasoning in a progressive strategy.

<p align="center"><img width="750" src=".github/method.jpg"></p>

## 🔥 News

- **`2024.03.28`** 🚀 VideoMind-2B is ready on [Hugging Face Spaces](https://huggingface.co/spaces/yeliudev/VideoMind-2B). Check it out!
- **`2024.03.21`** ⭐️ Code, model, and dataset release.
- **`2025.03.17`** 🎉 Our [tech report](https://arxiv.org/abs/2503.13444) is available online.

## 🕹️ Gradio Demo

https://github.com/user-attachments/assets/a4d99c05-aa73-4ed9-a275-2362a201bfec

Play with our [online demo](https://huggingface.co/spaces/yeliudev/VideoMind-2B) or see [DEMO.md](/docs/DEMO.md) for guidelines about how to deploy it locally.

## 📦 Datasets

We provide raw videos, compressed videos, and pre-processed annotations of **27 video grounding / QA datasets**, including our **VideoMind-SFT** (481K) for training and multiple benchmarks for evaluation. We also release the datasets used during our early exploration (but not included in the final version) to facilitate future research.

See our [dataset repo](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset) for more details.

## 🚀 Training

Our codebase supports training and evaluating on [27 video datasets and benchmarks](https://github.com/yeliudev/VideoMind/blob/main/videomind/dataset/sub_classes) with the following features.

- Flexible hardware settings: NVIDIA GPU / Ascend NPU, Single-Node / Multi-Node
- Efficient training techniques: DeepSpeed ZeRO, BF16, LoRA, SDPA, FlashAttention2, Liger-Kernel
- Customizing the base LLM and conversation templates
- Monitoring the training process via Tensorboard / Wandb
- Group sampling for mixed dataset training
- Multi-process evaluation on public benchmarks

See [TRAIN.md](/docs/TRAIN.md) for a quick start guide.

## 🔮 Evaluation

See [EVAL.md](/docs/EVAL.md) for details about evaluating VideoMind on public benchmarks.

## 📖 Citation

Please kindly cite our paper if you find this project helpful.

```bibtex
@article{liu2025videomind,
  title={VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning},
  author={Liu, Ye and Lin, Kevin Qinghong and Chen, Chang Wen and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2503.13444},
  year={2025}
}
```

[![Star History Chart](https://api.star-history.com/svg?repos=yeliudev/VideoMind&type=Timeline)](https://www.star-history.com/#yeliudev/VideoMind&Timeline)
