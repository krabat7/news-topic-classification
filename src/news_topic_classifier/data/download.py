from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import ClassLabel, DatasetDict, load_dataset

AG_NEWS_CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def _run_dvc_pull(repo_root: Path, target: Path) -> None:
    try:
        subprocess.run(
            ["dvc", "pull", str(target.as_posix())],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return


def download_data(dataset_name: str, out_dir: Path, text_joiner: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_name)

    def to_text(example: dict) -> dict:
        title = example.get("title", "")
        desc = example.get("description", "")
        text = example.get("text")
        if text is None:
            text = f"{title}{text_joiner}{desc}".strip()
        return {"text": text}

    def normalize_labels(split):
        cols = split.column_names
        if "label" in cols:
            label_key = "label"
        elif "class_index" in cols:
            label_key = "class_index"
        else:
            raise ValueError("Нет колонки label/class_index")

        split = split.rename_column(label_key, "labels")

        uniq = sorted(set(split["labels"]))
        if uniq == [1, 2, 3, 4]:
            split = split.map(lambda ex: {"labels": int(ex["labels"]) - 1})
        elif uniq == [0, 1, 2, 3]:
            pass
        else:
            raise ValueError(f"Неожиданные значения меток: {uniq[:20]}")

        class_label = ClassLabel(num_classes=4, names=AG_NEWS_CLASS_NAMES)
        split = split.cast_column("labels", class_label)
        return split

    train = normalize_labels(ds["train"].map(to_text))
    test = normalize_labels(ds["test"].map(to_text)) if "test" in ds else None

    dsd = DatasetDict({"train": train})
    if test is not None:
        dsd["test"] = test

    dsd.save_to_disk(str(out_dir))


def ensure_data(
    repo_root: Path,
    dataset_name: str,
    data_dir: Path,
    text_joiner: str,
) -> Path:
    _run_dvc_pull(repo_root, data_dir)

    if not data_dir.exists():
        download_data(dataset_name=dataset_name, out_dir=data_dir, text_joiner=text_joiner)

    return data_dir
