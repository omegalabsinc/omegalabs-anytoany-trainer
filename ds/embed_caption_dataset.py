from pathlib import Path
from itertools import chain
from functools import reduce
from operator import add

import torch
from torch.utils.data import IterableDataset


class SingleEmbedCaptionDataset(IterableDataset):
    def __init__(self, ib_embed_path, clip_embed_path, caption_path, world_size=1, rank=0):
        self.ib_embed_path = ib_embed_path
        self.clip_embed_path = clip_embed_path
        self.caption_path = caption_path
        self._world_size = world_size
        self._index = self._rank = rank
        self._len = None

    def __len__(self):
        if self._len is None:
            self._len = (len(torch.load(self.caption_path, map_location="cpu")) - self._rank) // self._world_size
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == self._rank:
            self.ib_embeds = torch.load(self.ib_embed_path, map_location="cpu")
            self.clip_embeds = torch.load(self.clip_embed_path, map_location="cpu")
            self.captions = torch.load(self.caption_path, map_location="cpu")
            assert self.ib_embeds.size(0) == self.clip_embeds.size(0), f'{self.ib_embed_path} len != {self.clip_embed_path} len'
            assert self.ib_embeds.size(0) == len(self.captions), f'{self.ib_embed_path} len({self.ib_embeds.size(0)}) != {self.caption_path} len({len(self.captions)})'

        if self._index >= self.ib_embeds.size(0):
            del self.ib_embeds, self.clip_embeds, self.captions
            raise StopIteration

        item = {
            "caption": self.captions[self._index],
            "ib_embed": self.ib_embeds[self._index],
            "clip_embed": self.clip_embeds[self._index],
        }
        self._index += self._world_size
        return item


class EmbedCaptionDataset(IterableDataset):
    def __init__(self, ib_embed_path, clip_embed_path, caption_path, world_size=1, rank=0):
        self.ib_embed_paths, self.clip_embed_paths, self.caption_paths = [
            sorted(list(Path('.').glob(p)))
            for p in [ib_embed_path, clip_embed_path, caption_path]
        ]
        assert len(self.ib_embed_paths) == len(self.clip_embed_paths)
        assert len(self.ib_embed_paths) == len(self.caption_paths)
        self._world_size = world_size
        self._rank = rank
        self._iter = chain.from_iterable(
            iter(SingleEmbedCaptionDataset(*paths, world_size=self._world_size, rank=self._rank))
            for paths in zip(self.ib_embed_paths, self.clip_embed_paths, self.caption_paths)
        )
        self._len = None

    def __len__(self):
        if self._len is None:
            self._len = reduce(
                add, [
                    len(SingleEmbedCaptionDataset(*paths, world_size=self._world_size, rank=self._rank))
                    for paths in zip(self.ib_embed_paths, self.clip_embed_paths, self.caption_paths)
                ]
            )
        return self._len

    def __iter__(self):
        return self._iter


if __name__ == "__main__":
    d = EmbedCaptionDataset(
        "ds/sam_llava/0[01].ib_embed.pt",
        "ds/sam_llava/0[01].clip_embed.pt",
        "ds/sam_llava/0[01].caption.pt",
    )
    for i, item in enumerate(d):
        if i % 1024 == 0: print(i)
