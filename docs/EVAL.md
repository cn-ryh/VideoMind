# Evaluating VideoMind

## 🛠️ Environment Setup

Please refer to [TRAIN.md](/docs/TRAIN.md) for setting up the environment.

## 📚 Checkpoint Preparation

Download the [base models](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d) and [VideoMind checkpoints](https://huggingface.co/collections/yeliudev/videomind-67dd41f42c57f0e7433afb36), and place them into the `model_zoo` folder.

```
VideoMind
└─ model_zoo
   ├─ Qwen2-VL-2B-Instruct
   ├─ Qwen2-VL-7B-Instruct
   ├─ VideoMind-2B
   ├─ VideoMind-7B
   └─ VideoMind-2B-FT-QVHighlights
```

## 📦 Dataset Preparation

Download the desired datasets / benchmarks from [Hugging Face](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset), extract the videos, and place them into the `data` folder. The processed files should be organized in the following structure (taking `charades_sta` as an example).

```
VideoMind
└─ data
   └─ charades_sta
      ├─ videos_3fps_480_noaudio
      ├─ durations.json
      ├─ charades_sta_train.txt
      └─ charades_sta_test.txt
```

## 🔮 Start Evaluation

### Multi-Process Inference (one GPU / NPU per process)

Use the following commands to evaluate VideoMind on different benchmarks. The default setting is to distribute the samples to 8 processes (each with one device) for acceleration. This mode requires at least 32GB memory per device.

```shell
# Evaluate VideoMind (2B / 7B) on benchmarks other than QVHighlights
bash scripts/evaluation/eval_auto_2b.sh <dataset> [<split>]
bash scripts/evaluation/eval_auto_7b.sh <dataset> [<split>]

# Evaluate VideoMind (2B) on QVHighlights
bash scripts/evaluation/eval_qvhighlights.sh
```

Here, `<dataset>` could be replaced with the following dataset names:

- Grounded VideoQA: `cgbench`, `rextime`, `nextgqa`, `qa_ego4d`
- Video Temporal Grounding: `charades_sta`, `activitynet_captions`, `tacos`, `ego4d_nlq`, `activitynet_rtl`
- General VideoQA: `videomme`, `mlvu`, `lvbench`, `mvbench`, `longvideobench`, `star`

The optional argument `<split>` could be `valid` or `test`, with `test` by default.

The inference outputs and evaluation metrics will be saved in the `outputs_2b` or `outputs_7b` folders by default.

### Multi-Device Inference (multiple GPUs / NPUs in one process)

You can also distribute the model to multiple devices to save memory. In this mode, only one process would be launched and the model is loaded into 8 devices.

```shell
bash scripts/evaluation/eval_dist_auto_2b.sh <dataset> [<split>]
bash scripts/evaluation/eval_dist_auto_7b.sh <dataset> [<split>]
```
