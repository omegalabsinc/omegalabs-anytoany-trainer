import os
from datasets import load_dataset, Dataset
import huggingface_hub
from huggingface_hub import HfApi
import ulid
import time
from omegaconf import OmegaConf, DictConfig
from tune_recipes.gen import InferenceRecipe


HF_DATASET = "omegalabsinc/omega-multimodal"
prefix = "default/train/"
suffix = ".parquet"
MAX_DATA_AGE = 4 * 60 * 60  # 4 hours
MAX_FILES = 8
MODEL_FILE_PREFIX = "meta_model"
CONFIG_FILE = "training_config.yml"
hf_api = HfApi()


def get_timestamp_from_filename(filename: str):
    return ulid.from_str(filename[len(prefix):filename.find(suffix)]).timestamp().timestamp


def pull_latest_omega_dataset() -> Dataset:
    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    recent_files = [
        f.rfilename
        for f in omega_ds_files if
        f.rfilename.startswith(prefix) and 
        time.time() - get_timestamp_from_filename(f.rfilename) < MAX_DATA_AGE
    ][:MAX_FILES]
    omega_dataset = load_dataset(HF_DATASET, data_files=recent_files)["train"]
    return omega_dataset


def load_ckpt_from_hf(hf_repo_id: str) -> InferenceRecipe:
    config_path = hf_api.hf_hub_download(repo_id=hf_repo_id, filename=CONFIG_FILE)
    ckpt_files = [f for f in hf_api.list_repo_files(repo_id=hf_repo_id) if f.startswith(MODEL_FILE_PREFIX)]
    if len(ckpt_files) == 0:
        raise ValueError(f"No checkpoint files found in {hf_repo_id}")
    ckpt_path = hf_api.hf_hub_download(repo_id=hf_repo_id, filename=ckpt_files[0])
    train_cfg = OmegaConf.load(config_path)
    train_cfg.model = DictConfig({
        "_component_": "models.mmllama3_8b",
        "use_clip": False,
        "perception_tokens": train_cfg.model.perception_tokens,
    })
    train_cfg.checkpointer.checkpoint_dir = os.path.dirname(ckpt_path)
    train_cfg.checkpointer.checkpoint_files = [os.path.basename(ckpt_path)]
    train_cfg.inference.max_new_tokens = 300
    inference_recipe = InferenceRecipe(train_cfg)
    inference_recipe.setup(cfg=train_cfg)
    return inference_recipe, train_cfg


def evaluate_checkpoint(inference_recipe: InferenceRecipe, config: DictConfig, dataset: Dataset):
    mini_batch = next(dataset.iter(batch_size=64))
    for video_emb, actual_caption in zip(mini_batch["video_embed"], mini_batch["description"]):
        generated_caption = inference_recipe.generate(cfg=config, video_ib_embed=video_emb)
        print("-----------------------------------")
        print(f"Actual caption: {actual_caption}")
        print(f"Generated caption: {generated_caption}")


def main():
    omega_dataset = pull_latest_omega_dataset()
    hf_repo_id = "salmanshahid/omega_a2a_test_2"
    model, cfg = load_ckpt_from_hf(hf_repo_id)
    ckpt_score = evaluate_checkpoint(model, cfg, omega_dataset)
    print(f"Checkpoint '{hf_repo_id}' score: {ckpt_score}")


if __name__ == "__main__":
    main()
