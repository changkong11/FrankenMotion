import os
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


def save_masked_stats(dataset, run_dir):
    is_training = dataset.is_training
    # don't drop anything
    dataset.is_training = False

    from tqdm import tqdm
    import torch
    from src.normalizer import Normalizer

    def _empty_stats():
        return {"sum": None, "sumsq": None, "count": None}

    def _update_masked_stats(stats, values, mask):
        values = values.detach().to(device="cpu", dtype=torch.float64)
        mask = mask.detach().to(device="cpu", dtype=torch.float64)
        masked_values = values * mask
        batch_sum = masked_values.sum(dim=0)
        batch_sumsq = (values.square() * mask).sum(dim=0)
        batch_count = mask.sum(dim=0)
        if stats["sum"] is None:
            stats["sum"] = batch_sum
            stats["sumsq"] = batch_sumsq
            stats["count"] = batch_count
        else:
            stats["sum"] += batch_sum
            stats["sumsq"] += batch_sumsq
            stats["count"] += batch_count

    def _update_unmasked_stats(stats, values):
        values = values.detach().to(device="cpu", dtype=torch.float64)
        values = values.reshape(-1, values.shape[-1])
        batch_sum = values.sum(dim=0)
        batch_sumsq = values.square().sum(dim=0)
        batch_count = torch.full_like(batch_sum, values.shape[0])
        if stats["sum"] is None:
            stats["sum"] = batch_sum
            stats["sumsq"] = batch_sumsq
            stats["count"] = batch_count
        else:
            stats["sum"] += batch_sum
            stats["sumsq"] += batch_sumsq
            stats["count"] += batch_count

    def _finalize_stats(stats):
        count = stats["count"]
        safe_count = torch.clamp(count, min=1)
        mean = stats["sum"] / safe_count
        numerator = stats["sumsq"] - stats["sum"].square() / safe_count
        numerator = torch.clamp(numerator, min=0)
        divisor = torch.clamp(count - 1, min=1)
        std = torch.sqrt(numerator / divisor)
        return mean.to(torch.float32), std.to(torch.float32)

    motion_stats_dir = os.path.join(run_dir, "motion_stats")
    os.makedirs(motion_stats_dir, exist_ok=True)

    text_stats_dir = os.path.join(run_dir, "text_stats")
    os.makedirs(text_stats_dir, exist_ok=True)

    motion_stats = _empty_stats()
    text_stats = _empty_stats()
    has_global_text = None

    try:
        logger.info("Compute motion/text embedding stats in streaming mode")

        for x in tqdm(dataset):
            _update_masked_stats(motion_stats, x["x"], x["stats_mask"])

            if has_global_text is None:
                # Global text (g_text=True) is a per-sequence embedding with no
                # frame-level mask; local per-part text is frame-aligned.
                has_global_text = "x" in x["tx"]

            if has_global_text:
                _update_unmasked_stats(text_stats, x["tx"]["x"])
            else:
                _update_masked_stats(
                    text_stats,
                    x["tx"]["local"]["x"],
                    x["tx"]["local"]["mask"],
                )

        mean_motionfeats, std_motionfeats = _finalize_stats(motion_stats)
        motion_normalizer = Normalizer(base_dir=motion_stats_dir, disable=True)
        motion_normalizer.save(mean_motionfeats, std_motionfeats)

        mean_textfeats, std_textfeats = _finalize_stats(text_stats)
        text_normalizer = Normalizer(base_dir=text_stats_dir, disable=True)
        text_normalizer.save(mean_textfeats, std_textfeats)

    finally:
        # re-enable dropping
        dataset.is_training = is_training


@hydra.main(config_path="configs", config_name="train_part", version_base="1.3")
def train(cfg: DictConfig):
    # Resuming if needed
    ckpt = None
    if cfg.resume_dir is not None:
        resume_dir = cfg.resume_dir
        max_epochs = cfg.trainer.max_epochs
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        cfg = read_config(cfg.resume_dir)
        cfg.trainer.max_epochs = max_epochs
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{resume_dir}")
    else:
        resume_dir = None
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")

    import src.prepare  # noqa
    import pytorch_lightning as pl

    pl.seed_everything(cfg.seed)

    logger.info("Loading the dataloaders")

    train_dataset = instantiate(cfg.data, split="train")
    val_dataset = instantiate(cfg.data, split="val")

    if resume_dir is None:
        logger.info("Computing statistics")
        save_masked_stats(train_dataset, cfg.run_dir)

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )

    logger.info("Loading the model")
    diffusion = instantiate(cfg.diffusion)

    logger.info("Training")
    trainer = instantiate(cfg.trainer)
    trainer.fit(diffusion, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
