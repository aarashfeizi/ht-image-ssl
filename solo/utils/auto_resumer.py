from argparse import ArgumentParser
import json
import os
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

from omegaconf import DictConfig
from solo.utils.misc import omegaconf_select

Checkpoint = namedtuple("Checkpoint", ["creation_time", "args", "checkpoint"])


class AutoResumer:
    SHOULD_MATCH = [
        "name",
        "backbone",
        "emb_model",
        "method",
        "augmentations",
        "data.dataset",
        "max_epochs",
        "optimizer.name",
        "data.num_nns",
        "data.num_nns_choice",
        "data.emb_path",
        "reload_freq",
        "optimizer.batch_size",
        "optimizer.lr",
        "optimizer.weight_decay",
        "wandb.project",
        "wandb.entity",
        "test",
        "pretrained_feature_extractor",
    ]

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = Path("trained_models"),
        max_hours: int = 36,
    ):
        """Autoresumer object that automatically tries to find a checkpoint
        that is as old as max_time.

        Args:
            checkpoint_dir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to "trained_models".
            max_hours (int): maximum elapsed hours to consider checkpoint as valid.
        """

        self.checkpoint_dir = checkpoint_dir
        self.max_hours = timedelta(hours=max_hours)

    @staticmethod
    def add_and_assert_specific_cfg(cfg: DictConfig) -> DictConfig:
        """Adds specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg.auto_resume = omegaconf_select(cfg, "auto_resume", default={})
        cfg.auto_resume.enabled = omegaconf_select(cfg, "auto_resume.enabled", default=False)
        cfg.auto_resume.max_hours = omegaconf_select(cfg, "auto_resume.max_hours", default=36)

        return cfg

    def find_checkpoint(self, cfg: DictConfig):
        """Finds a valid checkpoint that matches the arguments

        Args:
            cfg (DictConfig): DictConfig containing all settings of the model.
        """

        current_time = datetime.now()

        candidates = []
        for rootdir, _, files in os.walk(self.checkpoint_dir):
            rootdir = Path(rootdir)
            if files:
                # skip checkpoints that are empty
                try:
                    checkpoint_file = [rootdir / f for f in files if f.endswith(".ckpt")][0]
                except:
                    continue

                creation_time = datetime.fromtimestamp(os.path.getctime(checkpoint_file))
                if current_time - creation_time < self.max_hours:
                    ck = Checkpoint(
                        creation_time=creation_time,
                        args=rootdir / "args.json",
                        checkpoint=checkpoint_file,
                    )
                    candidates.append(ck)
        
        def filter_out_empty_key(v, key='kwargs'):
            if type(v) is DictConfig or type(v) is dict:
                if key in v.keys() and len(v[key]) == 0:
                    v.pop(key)
            return v

        if candidates:
            # sort by most recent
            candidates = sorted(candidates, key=lambda ck: ck.creation_time, reverse=True)

            for candidate in candidates:
                candidate_cfg = DictConfig(json.load(open(candidate.args)))
                if all(
                    filter_out_empty_key(omegaconf_select(candidate_cfg, param, None))
                    == filter_out_empty_key(omegaconf_select(cfg, param, None))
                    for param in AutoResumer.SHOULD_MATCH
                ):
                    wandb_run_id = getattr(candidate_cfg, "wandb_run_id", None)
                    return candidate.checkpoint, wandb_run_id

        return None, None
