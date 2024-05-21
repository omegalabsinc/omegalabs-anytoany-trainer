# OMEGA Labs Any-to-Any Trainer

This repo is meant as a companion to the [OMEGA Labs Bittensor Any-to-Any repo](https://github.com/omegalabsinc/omegalabs-bittensor-anytoany). It provides a starting point for training any-to-any models, beginning with video-captioning.

## Quickstart
To get started with training, just complete the following steps:
1. Build the docker container and run it: `make build-and-run`
(the following commands are to be run inside the `a2a` container)
2. Log into Huggingface: `huggingface-cli login`. Make sure your account has access to Llama-3-8B on HF, you can get access [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
3. Download the base model and datasets: `make download-everything`
4. Start training! `make finetune-x1` on a single GPU instance, or `make finetune-xN` where N is the number of GPUs e.g. `make finetune-x8` to train on a machine with 8 GPUs
