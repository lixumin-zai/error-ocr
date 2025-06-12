import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import os
from pathlib import Path

from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from PIL import ImageFilter, Image
from transformers import get_cosine_schedule_with_warmup
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from dataset import MyDataset
import wandb  # 添加wandb导入
from pytorch_lightning.loggers import WandbLogger


class TrainConfig:
    def __init__(self):
        self.model_path = "./model"

        self.learning_rate = 1e-5
        self.batch_size = 64
        self.num_training_samples_per_epoch = int(17000 / self.batch_size)
        self.warmup_steps = int(17000 / self.batch_size) 
        self.max_epochs = 10
        self.teacher_temp = 0.04 # softmax 温度系数更小，使得更突出
        self.student_temp = 0.1  
        self.center_momentum = 0.9
        self.ema_decay = 0.9995
        self.dataset_path = "./data/data"  # 替换为你自己的数据集路径
        self.num_workers = self.batch_size+1
        self.seed = 42
        self.max_length = 20

        self.save_path = "./model/save"


class TrocrLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(TrocrLightningModule, self).__init__()
        # 使用 torchvision 提供的预训练 vit_b_16
        self.config = config
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', cache_dir=self.config.model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten', cache_dir=self.config.model_path)
        self.model.config.decoder_start_token_id = self.processor.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

    def training_step(self, batch, batch_idx):
        images = [i[0] for i in batch]
        ground_truth_texts = [i[1] for i in batch]

        labels = self.processor.tokenizer(ground_truth_texts, 
            padding="max_length",
            truncation=True,  # 超过最大长度时进行截断
            max_length=self.config.max_length, 
            return_tensors="pt").input_ids.to(self.device)

        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss

        self.log('train_loss', loss, batch_size=self.config.batch_size,  prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = [i[0] for i in batch]
        ground_truth_texts = [i[1] for i in batch]
        labels = self.processor.tokenizer(ground_truth_texts, 
            padding="max_length",
            truncation=True,  # 超过最大长度时进行截断
            max_length=self.config.max_length, 
            return_tensors="pt").input_ids.to(self.device)

        # 图像预处理
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss

        self.log('val_loss', loss, batch_size=self.config.batch_size,  prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.05)
        return optimizer

    # def configure_optimizers(self):
    #     # 设置 warmup_steps 和 total_steps
    #     optimizer = optim.AdamW(self.student.parameters(), lr=self.config.learning_rate, weight_decay=0.05)
    #     scheduler = get_cosine_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=self.config.warmup_steps, 
    #         num_training_steps=self.config.num_training_samples_per_epoch*self.config.max_epochs
    #     )
    #     scheduler_config = {
    #         'scheduler': scheduler,
    #         'interval': 'step',  # 指定更新学习率的间隔是每步还是每个 epoch
    #         'frequency': 1,
    #         'name': 'learning_rate'
    #     }
    #     return [optimizer], [scheduler_config]

    def on_save_checkpoint(self, checkpoint):
        # 获取当前 epoch
        save_path = Path(self.config.save_path)
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)


def pil_collate_fn(temp):
    return temp

class DataPLModule(pl.LightningDataModule):
    def __init__(
        self, 
        dataset_path, 
        train_batch_size, 
        val_batch_size,
        num_workers, 
        seed
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size  
        self.val_batch_size = val_batch_size
        self.train_dataset = None 
        self.val_dataset = None 
        self.g = torch.Generator()
        self.g.manual_seed(seed)

    def setup(self, stage=None):
        self.train_dataset = MyDataset(
            self.dataset_path,
            "train"
        )
        self.val_dataset = MyDataset(
            self.dataset_path,
            "validation"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=pil_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=pil_collate_fn
        )

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


# 设置训练过程
if __name__ == '__main__':
    # 定义数据相关的配置
    
    train_config = TrainConfig()

    # 创建数据模块
    data_module = DataPLModule(
        dataset_path=train_config.dataset_path,
        train_batch_size=train_config.batch_size,
        val_batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        seed=train_config.seed
    )

    # 定义 ModelCheckpoint 回调，每个 epoch 保存一次模型
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",  # 模型保存的目录
        filename="{epoch:02d}-{train_acc:.2f}",  # 保存的文件名格式
        save_top_k=-1,  # 保存所有 epoch 的模型
        verbose=True,
        mode="min",  # 监控的 metric 越低越好
        save_weights_only=False,  # 保存整个模型而不仅仅是权重
        every_n_epochs=2 # 每个 epoch 保存一次
    )

    wandb_logger = WandbLogger(project="trocr", log_model=True)

    # 定义训练器
    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        strategy='ddp_find_unused_parameters_true',
        devices=[0,1],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],  # 添加 checkpoint 回调   
    )
    
    model = TrocrLightningModule(config=train_config)
    # 开始训练
    trainer.fit(model, data_module, 
        # ckpt_path="checkpoints/epoch=199-val_loss=0.02.ckpt"
    )
    wandb.finish()