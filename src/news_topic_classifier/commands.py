from __future__ import annotations

import sys
from pathlib import Path

import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from news_topic_classifier.data.datamodule import AGNewsDataModule
from news_topic_classifier.data.download import ensure_data
from news_topic_classifier.infer import infer_from_config
from news_topic_classifier.models.lightning_module import NewsClassifierModule
from news_topic_classifier.utils.git import get_git_commit_id
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


def train_from_config(overrides: list[str] | None = None) -> None:
    overrides = overrides or []
    repo_root = find_repo_root(Path(__file__).resolve())
    config_dir = repo_root / "configs"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=overrides)

    set_seed(int(cfg.seed))

    data_dir = repo_root / str(cfg.paths.data_dir)
    ensure_data(
        repo_root=repo_root,
        dataset_name=str(cfg.data.dataset_name),
        data_dir=data_dir,
        text_joiner=str(cfg.data.text_joiner),
    )

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
        data_dir=str(data_dir),
    )
    dm.setup()
    assert dm.vocab is not None

    model = NewsClassifierModule(
        vocab_size=dm.vocab.size,
        embedding_dim=int(cfg.model.embedding_dim),
        num_classes=int(cfg.model.num_classes),
        dropout=float(cfg.model.dropout),
        lr=float(cfg.lr),
    )

    mlflow_logger = MLFlowLogger(
        tracking_uri=str(cfg.logging.tracking_uri),
        experiment_name=str(cfg.logging.experiment_name),
        run_name=str(cfg.logging.run_name),
    )
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    mlflow_logger.log_hyperparams({"git_commit_id": get_git_commit_id()})

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(repo_root / "outputs" / "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best",
    )

    trainer = pl.Trainer(
        default_root_dir=str(repo_root / "outputs"),
        callbacks=[checkpoint_cb],
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=int(cfg.trainer.devices),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        enable_checkpointing=bool(cfg.trainer.enable_checkpointing),
        deterministic=bool(cfg.trainer.deterministic),
        logger=mlflow_logger,
    )

    trainer.fit(model, datamodule=dm)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: poetry run python -m news_topic_classifier.commands\n"
            "  <train|infer> [hydra_overrides...]"
        )

    command = sys.argv[1]
    overrides = sys.argv[2:]

    if command == "train":
        train_from_config(overrides)
        return

    if command == "infer":
        infer_from_config(overrides)
        return

    raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
