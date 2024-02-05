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
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from solo.args.pretrain import parse_cfg
from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.gps_dataset import GPS_Dataset_Wrapper
from solo.data.base_datamodule import BaseDataModule
import numpy as np
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

    if cfg.data.dataset.startswith('inat'):
        subsample_by = cfg.data.subsample_by
    else:
        subsample_by = 1


    if cfg.nn_augmentations is None:
        cfg.nn_augmentations = cfg.augmentations

    nn_augmentation = cfg.nn_augmentations[0]

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
            test=cfg.test,
            data_path=cfg.data.data_path
        )    
        
        train_dataset = misc.subsample_dataset(train_dataset, subsample_by=subsample_by)

        train_loader = prepare_dataloader(
            train_dataset, batch_size=cfg.optimizer.batch_size, num_workers=cfg.data.num_workers
        )

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint_config.dir, cfg.method),
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

    if cfg.checkpoint_config.enabled:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            cfg,
            logdir=os.path.join(cfg.checkpoint_config.dir, cfg.method),
            frequency=cfg.checkpoint_config.frequency,
            keep_prev=cfg.checkpoint_config.keep_prev,
        )
        callbacks.append(ckpt)




    if not cfg.test and cfg.es_factor > 1:
        print(f'Early stopping patience is {int(cfg.max_epochs // cfg.es_factor)} epochs!')
        early_stop_callback = EarlyStopping(monitor="val_acc1",
                                            min_delta=0.00,
                                            patience=int(cfg.max_epochs // cfg.es_factor),
                                            verbose=True, mode="max")
        callbacks.append(early_stop_callback)
    else:
        print(f'No Early Stopping!')

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
        tag_list = [cfg.data.dataset, cfg.method, f'{cfg.max_epochs}']
        if cfg.wandb.tags is not None:
            tag_list.extend(cfg.wandb.tags)
        if cfg.wandb.offline:
            tag_list.extend(['offline'])
        wandb_logger = WandbLogger(
            name=f'{cfg.name}_{unique_id}',
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            save_dir=cfg.wandb.save_dir,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
            tags=tag_list
        )

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    else:
        now = datetime.now()
        unique_id = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}_{now.microsecond}'
        tb_path = os.path.join(cfg.log_path, 'tensorboard')
        misc.make_dirs(tb_path)
        print(f'Running wandb exp {cfg.name}_{unique_id}')
        tb_logger = TensorBoardLogger(save_dir=tb_path,
                        name=f'{cfg.name}_{unique_id}')
        csv_logger = CSVLogger(save_dir=tb_path,
                        name=f'{cfg.name}_{unique_id}')

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    if cfg.checkpoint_config.enabled: # defining AFTER wandb_logger is definded
        file_name = f'{cfg.name}-' + '{epoch}-{val_acc1:.2f}'
        if wandb_logger.version is None:
            with open('.tmp.txt', 'r') as f:
                print('reading from .tmp.txt')
                logger_final_dir = f.read()
        else:
            logger_final_dir = wandb_logger.version if cfg.wandb.enabled else datetime.now().strftime('%Y%m%d%H%M%S_%f')
            print('saving to .tmp.txt')
            with open('.tmp.txt', 'w') as f:
                f.write(logger_final_dir)


        best_ckpt = ModelCheckpoint(
            monitor='val_acc1',
            verbose=True,
            save_last=False,
            save_top_k=1,
            mode='max',
            dirpath=os.path.join(cfg.checkpoint_config.dir, cfg.method, logger_final_dir),
            filename=file_name,
        )

        callbacks.append(best_ckpt)


    cache_path = os.path.join(cfg.log_path, 'cache')
    misc.make_dirs(cache_path)
    print('augs: ', cfg.augmentations)

    embeddings = None
    if not cfg.nnclr2:
    # Dataloader for embedding everything without shuffling and no transformations    

        emb_model_transform = build_no_transform(cfg.data.dataset, nn_augmentation)

        emb_train_dataset = prepare_datasets(
                cfg.data.dataset,
                emb_model_transform,
                train_data_path=cfg.data.train_path,
                data_format=cfg.data.format,
                no_labels=cfg.data.no_labels,
                data_fraction=cfg.data.fraction,
                test=cfg.test,
                data_path=cfg.data.data_path
            )

        emb_train_dataset = misc.subsample_dataset(emb_train_dataset, subsample_by=subsample_by)
            
        emb_train_loader = prepare_dataloader(emb_train_dataset, 
                                                        batch_size=cfg.optimizer.batch_size,
                                                        num_workers=cfg.data.num_workers,
                                                        shuffle=False,
                                                        drop_last=False)
        

    wandb_tbl = None
    
    if cfg.nnclr2:
        print('emb_model: ', cfg.emb_model)

        additional_str = ''
        if cfg.emb_model.train:
            additional_str += f'_ep{cfg.emb_model.epochs}_lr{cfg.emb_model.lr}_loss-{cfg.emb_model.loss}'

        if cfg.emb_model.pretrained == 'false':
            additional_str += f'_randomInit_seed{cfg.seed}'
        
        if cfg.emb_model.pretrained != 'true':
            additional_str += f'_{cfg.emb_model.pretrained}'
        else:
            additional_str += f'_imagenet'
        
        if cfg.data.subsample_by > 1 and cfg.data.dataset.startswith('inat'):
            additional_str += f'_SSB{cfg.data.subsample_by}'

        if cfg.emb_model.transform == 'noTransform':
            emb_model_transform = build_no_transform(cfg.data.dataset, nn_augmentation)
        else:
            if nn_augmentation.name != 'no_transform':
                additional_str += f'_{nn_augmentation.name}_seed{cfg.seed}'
            emb_model_transform = build_transform_pipeline(cfg.data.dataset, nn_augmentation)
        
        if cfg.data.dataset == 'aircrafts':
            if cfg.test:
                additional_str += '_TestMode'

        if cfg.emb_model.get_extended_features:
            output_string = ''.join(([str(i) for i in sorted(list(set(cfg.emb_model.outputs)))]))
            additional_str += f'_outputs{output_string}'

        embeddings_path = os.path.join(cache_path, f"{cfg.data.dataset}_{cfg.emb_model.name}{additional_str}_emb.npy")
        
        emb_train_dataset = prepare_datasets(
            cfg.data.dataset,
            emb_model_transform,
            train_data_path=cfg.data.train_path,
            data_format=cfg.data.format,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
            test=cfg.test,
            data_path=cfg.data.data_path
        )

        emb_train_dataset = misc.subsample_dataset(emb_train_dataset, subsample_by=subsample_by)
        
        emb_train_loader = prepare_dataloader(emb_train_dataset, 
                                                        batch_size=cfg.optimizer.batch_size,
                                                        num_workers=cfg.data.num_workers,
                                                        shuffle=False,
                                                        drop_last=False)

        using_presaved_embs = False 
        if cfg.data.emb_path is None or cfg.data.emb_path == '': # using_presaved_embs = False
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
                    emb_model = misc.train_emb_model(cfg, emb_model, emb_train_loader, cfg.emb_model.supervised)
                                
                emb_model.eval()
                if len(cfg.devices) > 1:
                    emb_model = torch.nn.DataParallel(emb_model, device_ids=cfg.devices)
                embeddings = misc.get_embeddings(emb_model, emb_train_loader)['embs']
                print('saving embeddings:')
                misc.save_npy(embeddings, embeddings_path)
                if cfg.data.dataset == 'pets':
                    dataset_data = emb_train_loader.dataset._images
                elif cfg.data.dataset == 'dtd':
                    dataset_data = emb_train_loader.dataset._image_files
                elif cfg.data.dataset == 'aircrafts':
                    dataset_data = emb_train_loader.dataset._image_files
                elif cfg.data.dataset == 'pathmnist' or cfg.data.dataset == 'tissuemnist':
                    dataset_data = emb_train_loader.dataset.imgs
                elif cfg.data.dataset == 'inat':
                    dataset_data = []
                    for cat_id, fname in emb_train_loader.dataset.index:
                        dataset_data.append(os.path.join(emb_train_loader.dataset.root,
                                                        emb_train_loader.dataset.all_categories[cat_id],
                                                        fname))
                elif cfg.data.dataset == 'inat18':
                    dataset_data = []
                    for fname in emb_train_loader.dataset.imgs:
                        dataset_data.append(os.path.join(emb_train_loader.dataset.root,
                                                        fname))
                elif cfg.data.dataset.startswith('hotel'):
                    imgs_lbls = emb_train_loader.dataset.imgs
                    dataset_data = list(list(zip(*imgs_lbls))[0])
                else:
                    dataset_data = emb_train_loader.dataset.data

                random_ids = misc.check_nns(embeddings, dataset_data, save_path=os.path.join(cache_path, f'nns_{cfg.data.dataset}_{cfg.emb_model.name}{additional_str}'), random_ids=cfg.emb_model.random_ids)
            else:
                print(f'Fetching {embeddings_path}')
                embeddings = misc.load_npy(embeddings_path)
        else: # cfg.data.emb_path is NOT None and NOT '' (empty)
            using_presaved_embs = True
            assert os.path.exists(os.path.join(cache_path, cfg.data.emb_path + '.npy'))
            embeddings_path = os.path.join(cache_path, cfg.data.emb_path + '.npy')
            print(f'Fetching {cfg.data.emb_path}')
            embeddings = misc.load_npy(embeddings_path).astype(np.float32)
            labels_name = f'{cfg.data.dataset}_test{cfg.test}_labels.npy'
            embeddings_lbls = np.load(os.path.join(cache_path, labels_name))

        
        print('Getting emb sim_matrix:')
        if using_presaved_embs:
            sim_matrix_path = os.path.join(cache_path, cfg.data.emb_path + '_sim_matrix.npy')
            dist_matrix_path = os.path.join(cache_path, cfg.data.emb_path + '_dist_matrix.npy')
            if not os.path.exists(sim_matrix_path) or not os.path.exists(dist_matrix_path):
                emb_dist_matrix, emb_sim_matrix = misc.get_sim_matrix(embeddings, gpu=torch.cuda.is_available())
                print(f'Caching sim_matrix and dist_matrix to {sim_matrix_path} and {dist_matrix_path}....')
                np.save(sim_matrix_path, emb_sim_matrix)
                np.save(dist_matrix_path, emb_dist_matrix)
            else:
                print(f'Using cached sim_matrix and dist_matrix from {sim_matrix_path} and {dist_matrix_path}!!! :)')
                emb_sim_matrix = np.load(sim_matrix_path)
                emb_dist_matrix = np.load(dist_matrix_path)
        else:
            emb_dist_matrix, emb_sim_matrix = misc.get_sim_matrix(embeddings, gpu=torch.cuda.is_available())

        print(f'num_nns: {cfg.data.num_nns}')
        print(f'num_nns_choice: {cfg.data.num_nns_choice}')
        
        extra_info = {}
        
        if cfg.data.threshold_mode == 'adaptive':
            threshold_k = cfg.data.threshold_k + 1 # default is 21
            if cfg.data.threshold_mode_type == 'mean+std':
                threshold = np.mean(emb_dist_matrix[:, 1:threshold_k]) + np.std(emb_dist_matrix[:, 1:threshold_k])
            elif cfg.data.threshold_mode_type == 'mean':
                threshold = np.mean(emb_dist_matrix[:, 1:threshold_k])
            elif cfg.data.threshold_mode_type == 'mean-std':
                threshold = np.mean(emb_dist_matrix[:, 1:threshold_k]) - np.std(emb_dist_matrix[:, 1:threshold_k])
            threshold_name = f'threshold_{cfg.data.threshold_mode}_{cfg.data.threshold_mode_type}'
            extra_info['emb_dist_AVG'] = np.mean(emb_dist_matrix[:, 1:threshold_k])
            extra_info['emb_dist_STD'] = np.std(emb_dist_matrix[:, 1:threshold_k])
            extra_info['emb_dist_VAR'] = np.var(emb_dist_matrix[:, 1:threshold_k])
            print(f'Seeting threshold to {threshold}')

        elif cfg.data.threshold_mode == 'fixed':
            threshold_name = f'threshold_{cfg.data.threshold_mode}'
            threshold = cfg.data.nn_threshold
            if threshold < 0:
                threshold_name = f'nnc_{cfg.data.num_nns_choice}'


        clust_dist, clust_lbls = None, None
        if cfg.data.clustering_algo == 'kmeans':
            if cfg.data.num_clusters > 1:
                clusters = misc.get_clusters(embeddings, k=cfg.data.num_clusters, gpu=torch.cuda.is_available())
                clust_dist = clusters['dist']
                clust_lbls = clusters['lbls']
        
        elif cfg.data.clustering_algo.startswith('louvain'):
            clust_dist = None
            if cfg.data.clustering_algo == 'louvainW':
                clust_lbls, knn_graph = misc.get_louvain_clusters_weighted(emb_sim_matrix, dist_matrix=emb_dist_matrix, seed=cfg.seed, threshold=threshold)
            elif cfg.data.clustering_algo == 'louvainU':
                clust_lbls, knn_graph = misc.get_louvain_clusters_unweighted(emb_sim_matrix, dist_matrix=emb_dist_matrix, seed=cfg.seed, k=cfg.data.num_nns_choice + 1)
        
        assert len(train_dataset) == len(embeddings)
        
        
        if cfg.data.clustering_algo.startswith('louvain'):
            extra_info['no_clusters'] = len(set(clust_lbls))
        
        if cfg.data.plot_distances or cfg.data.plot_distances_after_epoch:
            plot_save_path = os.path.join(cfg.log_path, 'pos_neg_histograms_new/', f'{cfg.data.dataset}_n{cfg.method}_{threshold_name}_reloadFreq{cfg.data.reload_freq}')
            misc.make_dirs(plot_save_path)
        else:
            plot_save_path = './'

        train_dataset = GPS_Dataset_Wrapper(dataset=train_dataset,
                                               dataset_name=cfg.data.dataset,
                                                sim_matrix=emb_sim_matrix,
                                                dist_matrix=emb_dist_matrix,
                                                cluster_lbls=clust_lbls,
                                                nn_threshold=threshold,
                                                num_nns=cfg.data.num_nns,
                                                num_nns_choice=cfg.data.num_nns_choice,
                                                filter_sim_matrix=cfg.data.filter_sim_matrix,
                                                subsample_by=1,
                                                clustering_algo=cfg.data.clustering_algo,
                                                extra_info=extra_info,
                                                plot_distances=False,
                                                save_path=plot_save_path,
                                                no_reloads=1,
                                                hop=cfg.data.nn_hop)
        
        if cfg.data.plot_distances_after_epoch:
            plot_embeddings_cb = misc.PlotEmbeddingsCallback(save_path=plot_save_path, dataset_name=cfg.data.dataset, data_loader=emb_train_loader)
            callbacks.append(plot_embeddings_cb)
        
        if cfg.wandb.plot_embs:
            freq = cfg.wandb.plot_embs_freq
            num_per_class = cfg.wandb.plot_embs_per_class
            if freq == 0:
                freq = cfg.max_epochs
            
            num_of_classes = 10
            print(f'** wandb ** plotting {num_per_class} embeddings of {num_of_classes} classes every {freq} epochs! :) ')
            plot_embeddings_wb_cb = misc.PlotEmbeddingsWandBCallback(freq=freq,
                                                                  num_of_classes=num_of_classes,
                                                                  max_epochs=cfg.max_epochs,
                                                                  data_loader=emb_train_loader,
                                                                  labels_to_use=cfg.wandb.labels_to_use,
                                                                  downsampe_embs=True,
                                                                  num_per_class=num_per_class)
            callbacks.append(plot_embeddings_wb_cb)

        print('Relevant class percentage: ', train_dataset.relevant_classes)
        print('Not from cluster percentage: ', train_dataset.not_from_cluster_percentage)
        print('Number of nns: ', train_dataset.no_nns)
        class_percentage_cb = misc.ClassNNPecentageCallback(dataset_name=cfg.data.dataset, 
                                                            data_loader=emb_train_loader, 
                                                            save_path=plot_save_path, 
                                                            plot_nearest_neighbors=cfg.data.plot_distances,
                                                            key=cfg.nn_key)
        callbacks.append(class_percentage_cb)

        train_loader = prepare_dataloader(
            train_dataset, batch_size=cfg.optimizer.batch_size, num_workers=cfg.data.num_workers
        )

    elif cfg.method == 'nnclr':
        if cfg.data.plot_distances:
            plot_save_path = os.path.join(cfg.log_path, 'pos_neg_histograms_new/', f'{cfg.data.dataset}_{cfg.method}')
            misc.make_dirs(plot_save_path)
        else:
            plot_save_path = './'

        if cfg.data.plot_distances_after_epoch:
            no_shuffle_train_loader = prepare_dataloader(
                                            train_dataset, 
                                            batch_size=cfg.optimizer.batch_size, 
                                            num_workers=cfg.data.num_workers, 
                                            shuffle=False, 
                                            drop_last=False)
            plot_embeddings_cb = misc.PlotEmbeddingsCallback(save_path=plot_save_path, 
                                                             dataset_name=cfg.data.dataset,
                                                             data_loader=no_shuffle_train_loader, 
                                                             index=0, 
                                                             key='z')
            callbacks.append(plot_embeddings_cb)

        # class_percentage_cb = misc.ClassNNPecentageCallback_NNCLR(dataset_name=cfg.data.dataset, save_path=plot_save_path)
        # callbacks.append(class_percentage_cb)
    
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
            subsample_by=subsample_by,
            test=cfg.test,
            min_scale_224=(cfg.backbone.name.startswith('vit')) or ('_clip_' in cfg.backbone.name),
            data_path=cfg.data.data_path
        )

    datamodule = BaseDataModule(model=model,
                                filter_sim_matrix=cfg.data.filter_sim_matrix,
                                subsample_by=1,
                                num_clusters=cfg.data.num_clusters,
                                nn_threshold=cfg.data.nn_threshold,
                                threshold_mode=cfg.data.threshold_mode,
                                clustering_algo=cfg.data.clustering_algo,
                                seed=cfg.seed,
                                threshold_k=cfg.data.threshold_k,
                                key=cfg.nn_key)
    
    datamodule.set_emb_dataloder(emb_train_loader)
    datamodule.set_train_loader(train_loader)
    datamodule.set_val_loader(val_loader)

    if cfg.wandb.enabled:
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
        if wandb_logger._offline:
            misc.handle_wandb_offline(wandb_logger=wandb_logger)
        
        if embeddings is not None:
            wandb_tbl, mask_n_labels_used = misc.get_wandb_table(embeddings=embeddings,
                                        embedding_labels=embeddings_lbls, labels_to_use=cfg.wandb.labels_to_use)
        

            if wandb_tbl is not None and cfg.wandb.plot_embs and cfg.data.num_nns_choice > 1:
                wandb_logger.log_metrics({'gps-embeddings': wandb_tbl})
    else:
        tb_logger.log_hyperparams(OmegaConf.to_container(cfg))
        csv_logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else [csv_logger, tb_logger],
            "callbacks": callbacks,
            "enable_checkpointing": cfg.checkpoint_config.enabled,
            "reload_dataloaders_every_n_epochs": cfg.data.reload_freq,
            "log_every_n_steps": 500,
            #"progress_bar_refresh_rate": 0, # turn off progress bar
            "strategy": DDPStrategy(find_unused_parameters=(cfg.method != 'mae')) if cfg.strategy == "ddp" else cfg.strategy,
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

    if cfg.method == 'mae':
        batch_size = cfg.optimizer.batch_size
        model.cpu()
        backbone = model.backbone.cuda()
        log_batchsize = int(np.log2(batch_size))
        for _ in range(log_batchsize):
            emb_train_loader = prepare_dataloader(emb_train_dataset, 
                                batch_size=batch_size,
                                num_workers=cfg.data.num_workers,
                                shuffle=False,
                                drop_last=False)
            try:
                print(f'Trying batch size: {batch_size}')
                out = backbone(next(iter(emb_train_loader))[1].cuda())
                print(f'Batch size {batch_size} works!')

            except:
                batch_size = batch_size // 2
                print(f'Batch size too big, trying {batch_size}')


        embeddings, embedding_lbls = misc.get_mae_embeddings(backbone, emb_train_loader, device='cuda', lbls=True)

        full_path = os.path.join(cache_path, f'{cfg.data.dataset}_mae_{cfg.backbone.name}_eps{cfg.max_epochs}_clr{cfg.optimizer.classifier_lr}_lr{cfg.optimizer.lr}_wd{cfg.optimizer.weight_decay}_mr{cfg.method_kwargs.mask_ratio}.npy')

        np.save(full_path, embeddings)

        emb_tbl, _ = misc.get_wandb_table(embeddings=embeddings, 
                                       embedding_labels=embedding_lbls, labels_to_use=cfg.wandb.labels_to_use)
        wandb_logger.log_metrics({f'MAE-Embeddings at epoch {cfg.max_epochs}': emb_tbl})

if __name__ == "__main__":
    main()