from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def size(self) -> int:
        return len(self.itos)


def simple_tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_vocab(texts: Iterable[str], max_vocab_size: int, min_freq: int) -> Vocab:
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))

    itos = ["<pad>", "<unk>"]
    for token, freq in counter.most_common(max_vocab_size - len(itos)):
        if freq < min_freq:
            break
        itos.append(token)

    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def encode(text: str, vocab: Vocab, max_length: int) -> tuple[list[int], list[int]]:
    tokens = simple_tokenize(text)[:max_length]
    ids = [vocab.stoi.get(tok, 1) for tok in tokens]
    mask = [1] * len(ids)
    return ids, mask


def collate_batch(batch: list[dict]) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        ids = x["input_ids"]
        mask = x["attention_mask"]
        pad = max_len - len(ids)
        input_ids.append(ids + [0] * pad)
        attention_mask.append(mask + [0] * pad)
        labels.append(x["labels"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class AGNewsDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        text_joiner: str,
        val_size: float,
        batch_size: int,
        num_workers: int,
        max_vocab_size: int,
        min_freq: int,
        max_length: int,
        seed: int,
        data_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.text_joiner = text_joiner
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.max_length = max_length
        self.seed = seed
        self.data_dir = data_dir

        self.vocab: Vocab | None = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _load(self):
        if self.data_dir is not None:
            p = Path(self.data_dir)
            if not p.exists():
                raise FileNotFoundError(
                    f"Data dir not found: {p}. "
                    "Run train/infer which should call ensure_data() (DVC pull / download)."
                )
            return load_from_disk(str(p))
        return load_dataset(self.dataset_name)

    def setup(self, stage: str | None = None) -> None:
        ds = self._load()
        train = ds["train"]

        def to_text(example: dict) -> dict:
            title = example.get("title", "")
            desc = example.get("description", "")
            text = example.get("text")
            if text is None:
                text = f"{title}{self.text_joiner}{desc}".strip()
            return {"text": text}

        if "text" not in train.column_names:
            train = train.map(to_text)

        if "labels" not in train.column_names:
            if "label" in train.column_names:
                train = train.rename_column("label", "labels")
            elif "class_index" in train.column_names:
                train = train.rename_column("class_index", "labels")
            else:
                raise ValueError(f"No labels column found. Columns: {train.column_names}")

        uniq = sorted(set(train["labels"]))
        if uniq == [1, 2, 3, 4]:
            train = train.map(lambda ex: {"labels": int(ex["labels"]) - 1})

        try:
            split = train.train_test_split(
                test_size=self.val_size,
                seed=self.seed,
                stratify_by_column="labels",
            )
            train_ds = split["train"]
            val_ds = split["test"]
        except Exception:
            split = train.train_test_split(test_size=self.val_size, seed=self.seed)
            train_ds = split["train"]
            val_ds = split["test"]

        self.vocab = build_vocab(
            texts=train_ds["text"],
            max_vocab_size=self.max_vocab_size,
            min_freq=self.min_freq,
        )

        def to_ids(example: dict) -> dict:
            ids, mask = encode(example["text"], self.vocab, self.max_length)
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "labels": int(example["labels"]),
            }

        self.train_ds = train_ds.map(to_ids, remove_columns=train_ds.column_names)
        self.val_ds = val_ds.map(to_ids, remove_columns=val_ds.column_names)

        if "test" in ds:
            test = ds["test"]
            if "text" not in test.column_names:
                test = test.map(to_text)

            if "labels" not in test.column_names:
                if "label" in test.column_names:
                    test = test.rename_column("label", "labels")
                elif "class_index" in test.column_names:
                    test = test.rename_column("class_index", "labels")

            uniq_test = sorted(set(test["labels"]))
            if uniq_test == [1, 2, 3, 4]:
                test = test.map(lambda ex: {"labels": int(ex["labels"]) - 1})

            self.test_ds = test.map(to_ids, remove_columns=test.column_names)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("Test dataset is not available.")
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            persistent_workers=self.num_workers > 0,
        )
