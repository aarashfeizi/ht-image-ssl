# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import inspect
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from solo.args.pretrain import parse_cfg
from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.nnclr2_dataset import NNCLR2_Dataset_Wrapper
from solo.data.base_datamodule import BaseDataModule
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
    build_no_transform,
)
from solo.methods import METHODS
from solo.emb_methods import EMB_METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils import misc
from datetime import datetime


try:
    from solo.data.dali_dataloader import PretrainDALIDataModule, build_transform_pipeline_dali
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    # args = get_args()
    cfg = parse_cfg(cfg)

    seed_everything(cfg.seed)

    assert cfg.method in METHODS, f"Choose from {METHODS.keys()}"


    # pretrain dataloader
    if cfg.data.format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline_dali(
                        cfg.data.dataset, aug_cfg, dali_device=cfg.dali.device
                    ),
                    aug_cfg.num_crops,
                )
            )
        transform = FullTransformPipeline(pipelines)

        dali_datamodule = PretrainDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            transforms=transform,
            num_large_crops=cfg.data.num_large_crops,
            num_small_crops=cfg.data.num_small_crops,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
            dali_device=cfg.dali.device,
            encode_indexes_into_labels=cfg.dali.encode_indexes_into_labels,
        )
        dali_datamodule.val_dataloader = lambda: val_loader
    else:
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline(cfg.data.dataset, aug_cfg), aug_cfg.num_crops
                )
            )
        transform = FullTransformPipeline(pipelines)

        if cfg.debug_augmentations:
            print("Transforms:")
            print(transform)

        train_dataset = prepare_datasets(
            cfg.data.dataset,
            transform,
            train_data_path=cfg.data.train_path,
            data_format=cfg.data.format,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
        )    

        train_loader = prepare_dataloader(
            train_dataset, batch_size=cfg.optimizer.batch_size, num_workers=cfg.data.num_workers
        )

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, cfg.method),
            max_hours=cfg.auto_resume.max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint

    callbacks = []

    if cfg.checkpoint.enabled:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            cfg,
            logdir=os.path.join(cfg.checkpoint.dir, cfg.method),
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
        )
        callbacks.append(ckpt)

    if cfg.auto_umap.enabled:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            cfg.name,
            logdir=os.path.join(cfg.auto_umap.dir, cfg.method),
            frequency=cfg.auto_umap.frequency,
        )
        callbacks.append(auto_umap)

    # wandb logging
    if cfg.wandb.enabled:
        now = datetime.now()
        unique_id = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}_{now.microsecond}'
        print(f'Running wandb exp {cfg.name}_{unique_id}')
        wandb_logger = WandbLogger(
            name=f'{cfg.name}_{unique_id}',
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            save_dir=cfg.wandb.save_dir,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)



    cache_path = os.path.join(cfg.log_path, 'cache')
    misc.make_dirs(cache_path)
    print('augs: ', cfg.augmentations)
    emb_train_loader = None
    if cfg.nnclr2:
        print('emb_model: ', cfg.emb_model)
        additional_str = ''
        if cfg.emb_model.train:
            additional_str += f'_ep{cfg.emb_model.epochs}_lr{cfg.emb_model.lr}_loss-{cfg.emb_model.loss}'

        if cfg.emb_model.pretrained != 'true':
            additional_str += f'_randomInit_seed{cfg.seed}'


        embeddings_path = os.path.join(cache_path, f"{cfg.data.dataset}_{cfg.emb_model.name}{additional_str}_emb.npy")

        if cfg.emb_model.transform == 'noTransform':
            emb_model_transform = build_no_transform(cfg.data.dataset, cfg.augmentations[0])
        else:
            emb_model_transform = build_transform_pipeline(cfg.data.dataset, cfg.augmentations[0])
        emb_train_dataset = prepare_datasets(
            cfg.data.dataset,
            emb_model_transform,
            train_data_path=cfg.data.train_path,
            data_format=cfg.data.format,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
        )
        
        emb_train_loader = prepare_dataloader(emb_train_dataset, 
                                                        batch_size=cfg.optimizer.batch_size,
                                                        num_workers=cfg.data.num_workers,
                                                        shuffle=False,
                                                        drop_last=False)


        if not os.path.exists(embeddings_path):
            print(f'Creating {embeddings_path}')
            if cfg.emb_model.name.startswith('autoencoder'):
                model_type = 'autoencoder'
            elif cfg.emb_model.name.startswith('conv_autoencoder'):
                model_type = 'conv_autoencoder'
            elif cfg.emb_model.name.startswith('resnet'):
                model_type = 'resnet'
            emb_model = EMB_METHODS[model_type](cfg)

            emb_model.cuda()
            
            if cfg.emb_model.train:
                emb_model.train()
                print('Start training emb_model')
                emb_model = misc.train_emb_model(cfg, emb_model, emb_train_loader)
                            
            emb_model.eval()
            embeddings = misc.get_embeddings(emb_model, emb_train_loader)
            misc.save_npy(embeddings, embeddings_path)
            if cfg.data.dataset == 'pets':
                dataset_data = emb_train_loader.dataset._images
            elif cfg.data.dataset == 'dtd':
                dataset_data = emb_train_loader.dataset._image_files
            else:
                dataset_data = emb_train_loader.dataset.data

            random_ids = misc.check_nns(embeddings, dataset_data, save_path=os.path.join(cache_path, f'nns_{cfg.data.dataset}_{cfg.emb_model.name}{additional_str}'), random_ids=cfg.emb_model.random_ids)
        else:
            print(f'Fetching {embeddings_path}')
            embeddings = misc.load_npy(embeddings_path)

        emb_dist_matrix, emb_sim_matrix = misc.get_sim_matrix(embeddings, gpu=torch.cuda.is_available())
        assert len(train_dataset) == len(embeddings)

        if cfg.nnclr2:
            print(f'num_nns: {cfg.data.num_nns}')
            print(f'num_nns_choice: {cfg.data.num_nns_choice}')
            train_dataset = NNCLR2_Dataset_Wrapper(dataset=train_dataset,
                                                    sim_matrix=emb_sim_matrix,
                                                    num_nns=cfg.data.num_nns,
                                                    num_nns_choice=cfg.data.num_nns_choice,
                                                    filter_sim_matrix=cfg.data.filter_sim_matrix)

            train_loader = prepare_dataloader(
                train_dataset, batch_size=cfg.optimizer.batch_size, num_workers=cfg.data.num_workers
            )


    model = METHODS[cfg.method](cfg)
    misc.make_contiguous(model)
    # can provide up to ~20% speed up
    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)

    # validation dataloader for when it is available
    if cfg.data.dataset == "custom" and (cfg.data.no_labels or cfg.data.val_path is None):
        val_loader = None
    elif cfg.data.dataset in ["imagenet100", "imagenet"] and cfg.data.val_path is None:
        val_loader = None
    else:
        if cfg.data.format == "dali":
            val_data_format = "image_folder"
        else:
            val_data_format = cfg.data.format

        _, val_loader = prepare_data_classification(
            cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            data_format=val_data_format,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
        )

    datamodule = BaseDataModule(model=model)
    datamodule.set_emb_dataloder(emb_train_loader)
    datamodule.set_train_loader(train_loader)
    datamodule.set_val_loader(val_loader)

    if cfg.wandb.enabled:
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "reload_dataloaders_every_n_epochs": cfg.data.reload_freq,
            "progress_bar_refresh_rate": 0, # turn off progress bar
            "strategy": DDPStrategy(find_unused_parameters=True) if cfg.strategy == "ddp" else cfg.strategy,
        }
    )
    trainer = Trainer(**trainer_kwargs)

    # fix for incompatibility with nvidia-dali and pytorch lightning
    # with dali 1.15 (this will be fixed on 1.16)
    # https://github.com/Lightning-AI/lightning/issues/12956
    try:
        from pytorch_lightning.loops import FitLoop

        class WorkaroundFitLoop(FitLoop):
            @property
            def prefetch_batches(self) -> int:
                return 1

        trainer.fit_loop = WorkaroundFitLoop(
            trainer.fit_loop.min_epochs, trainer.fit_loop.max_epochs
        )
    except:
        pass

    if cfg.data.format == "dali":
        trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
