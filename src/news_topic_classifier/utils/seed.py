import pytorch_lightning as pl


def set_seed(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
