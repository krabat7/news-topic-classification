from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from news_topic_classifier.data.datamodule import AGNewsDataModule, encode, load_vocab
from news_topic_classifier.data.download import ensure_data
from news_topic_classifier.models.lightning_module import NewsClassifierModule
from news_topic_classifier.utils.seed import set_seed


def find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(10):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError("Repo root not found (could not locate .git directory)")


def _to_tensor_batch(token_ids: list[list[int]], max_len: int) -> dict[str, torch.Tensor]:
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    for ids in token_ids:
        ids = ids[:max_len]
        mask = [1] * len(ids)
        pad = max_len - len(ids)
        input_ids.append(ids + [0] * pad)
        attention_mask.append(mask + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
    }


def infer_from_config(overrides: list[str] | None = None) -> None:
    overrides = overrides or []

    repo_root = find_repo_root(Path(__file__).resolve())
    config_dir = repo_root / "configs"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="infer", overrides=overrides)

    set_seed(int(cfg.seed))

    data_dir = repo_root / str(cfg.paths.data_dir)
    vocab_path = repo_root / str(cfg.paths.vocab_path)
    ensure_data(
        repo_root=repo_root,
        dataset_name=str(cfg.data.dataset_name),
        data_dir=data_dir,
        text_joiner=str(cfg.data.text_joiner),
    )

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found at {vocab_path}. Run train first.")
    vocab = load_vocab(vocab_path)

    dm = AGNewsDataModule(
        dataset_name=str(cfg.data.dataset_name),
        text_joiner=str(cfg.data.text_joiner),
        val_size=float(cfg.data.val_size),
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.data.num_workers),
        max_vocab_size=int(cfg.data.max_vocab_size),
        min_freq=int(cfg.data.min_freq),
        max_length=int(cfg.max_length),
        seed=int(cfg.seed),
        data_dir=str((repo_root / str(cfg.paths.data_dir)).as_posix()),
        vocab=vocab,
    )
    dm.setup()
    if dm.vocab is None:
        raise RuntimeError("Vocab was not built (dm.vocab is None).")

    ckpt_path = Path(str(cfg.infer.ckpt_path))
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = NewsClassifierModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        vocab_size=dm.vocab.size,
        embedding_dim=int(cfg.model.embedding_dim),
        num_classes=int(cfg.model.num_classes),
        dropout=float(cfg.model.dropout),
        lr=float(cfg.lr),
    )
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and bool(cfg.infer.use_cuda) else "cpu"
    )
    model.to(device)

    texts = list(cfg.infer.texts)
    if len(texts) == 0:
        raise ValueError("configs/infer.yaml: infer.texts is empty")

    token_ids: list[list[int]] = []
    for t in texts:
        ids, _ = encode(str(t), dm.vocab, int(cfg.max_length))
        token_ids.append(ids)

    batch = _to_tensor_batch(token_ids, max_len=int(cfg.max_length))
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        logits = model.model(batch["input_ids"], batch["attention_mask"])
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().tolist()
        confs = torch.max(probs, dim=-1).values.cpu().tolist()

    labels = list(cfg.infer.class_names)

    for text, pred, conf in zip(texts, preds, confs, strict=True):
        name = labels[pred] if 0 <= pred < len(labels) else str(pred)
        print(f"TEXT: {text}")
        print(f"PRED: {pred} ({name}), CONF: {conf:.4f}")
        print("-" * 60)

    if bool(cfg.infer.print_config):
        print("\nResolved config:")
        print(OmegaConf.to_yaml(cfg, resolve=True))
