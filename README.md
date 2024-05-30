# OMEGA Labs Any-to-Any Trainer

---
### NOTE: This repository is being archived. All the model training code has been merged into the [OMEGA Labs Any-to-Any Bittensor](https://github.com/omegalabsinc/omegalabs-anytoany-bittensor) repo. Please use that one instead.

---

This repo is meant as a companion to the [OMEGA Labs Any-to-Any Bittensor repo](https://github.com/omegalabsinc/omegalabs-anytoany-bittensor). It provides a starting point for training any-to-any models, beginning with video-captioning.

## Quickstart
To get started with training, just complete the following steps:
0. Make sure to review the [requirements](#requirements)!
1. Build the docker container and run it: `make build-and-run`
(the following commands are to be run inside the `a2a` container)
2. Log into Huggingface: `huggingface-cli login`. Make sure your account has access to Llama-3-8B on HF, you can get access [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
3. Download the base model and datasets: `make download-everything`
4. Start training! `make finetune-x1` on a single GPU instance, or `make finetune-xN` where N is the number of GPUs e.g. `make finetune-x8` to train on a machine with 8 GPUs
5. **Important**: Once you are done training, don't forget to upload the model! [See instructions below.](#uploading-model-to-huggingface)

## Uploading model to Huggingface
From within the container, just run `python upload_ckpt_hf.py --ckpt_dir <checkpoint directory> --epoch <epoch number to upload> --hf_repo_id <repo id to upload to>`

## Requirements
- GPU with at least 48 GB VRAM
- CPU with at least 40 GB RAM

## Experiment Ideas
Some potential ways for miners to train better checkpoints and get an edge in the incentives are:
- Experiment with the perception_tokens hyperparameter (this refers to how many text tokens each image/audio/video is mapped to)
- Incorporate new datasets and experiment with data cleaning / filtering techniques
- Tweak the prompt templating to make the model robust towards more generic instructions
- Bring in more multi-modal interleaved datasets (e.g. datasets where images, video, audio, and text all appear in-line)
  - Could try synthetic data generation here

## Future Experiment Ideas
Miners cannot experiment with the following ideas presently because the validation mechanism is intentionally fairly restrictive to start out, in order to limit the experimentation space. However, these are good directions to be thinking about for future experiments:
- Multimodal tokenization (early fusion): [Chameleon](https://arxiv.org/abs/2405.09818)
  - In general, we believe there will be a big push towards training end-to-end sequences 
- [JEPA's](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
- High-fidelity auto-regressive video generation
- Zero-shot personalization
- Screen understanding models
