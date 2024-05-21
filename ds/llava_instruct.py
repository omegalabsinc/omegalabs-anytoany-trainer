from typing import Any, Tuple, List, Mapping
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    Message,
    validate_messages,
)
import numpy as np

from models.tokenizer import START_IMAGE, END_IMAGE
from ds.embed_caption_dataset import EmbedCaptionDataset


class LlavaInstructDataset(IterableDataset):
    def __init__(self, ib_embed_path, clip_embed_path, caption_path, tokenizer, train_on_input=False, max_seq_len=512, world_size=1, rank=0):
        self._data = EmbedCaptionDataset(ib_embed_path, clip_embed_path, caption_path, world_size, rank)
        self._len = len(self._data)
        self._data = iter(self._data)
        self._tokenizer = tokenizer
        self._image_ids = self._tokenizer.encode(START_IMAGE + END_IMAGE, add_eos=False, add_bos=False)
        self.train_on_input = train_on_input
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        return self._prepare_sample(next(self._data))

    def _turn_to_message(self, turn: Mapping[str, Any]) -> Message:
        role = {"human": "user", "gpt": "assistant"}[turn["from"]]
        return Message(
            role=role,
            content=turn["value"].replace("<image>", f"{START_IMAGE}0{END_IMAGE}"),
            masked=(role == "user" and not self.train_on_input)
        )

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        messages = [self._turn_to_message(turn) for turn in sample["caption"]]
        validate_messages(messages)

        tokens, mask = self._tokenizer.tokenize_messages(
            messages, max_seq_len=self.max_seq_len
        )
        if not self.train_on_input:
            # don't learn that "<|start_header_id|>assistant" always comes after <|eot_id|>
            try:
                first_false_idx = mask.index(True)
                mask[first_false_idx:first_false_idx+3] = [False, False, False]
            except ValueError:
                pass
        mask[-1] = False # also learn <|eot_id|>

        # ensure within-image tags tokens are masked out
        mask = np.array(mask)
        image_embed_mask_ids = []
        context = {}
        in_image_embed = False
        for idx, tok in enumerate(tokens):
            in_image_embed = in_image_embed and not tok == self._image_ids[1]
            if in_image_embed:
                image_embed_mask_ids.append(idx)
                #tokens[idx] # to support multiple embeds: get the value, match it up with the sample embed
                context[idx] = {k: sample[k] for k in ["ib_embed", "clip_embed"]}
            in_image_embed = in_image_embed or tok == self._image_ids[0]
        if image_embed_mask_ids:
            mask[image_embed_mask_ids] = True

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels)

        return tokens, labels, context


