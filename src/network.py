# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# Custom packages
from src.metric import MyF1Score, MyAccuracy
import src.config as cfg
from src.util import show_setting
from src.networks.MyAlexNet import MyAlexNet
from src.networks.MyConvNeXtV2 import MyConvNeXtV2
from src.networks.MyEfficientNetL2 import MyEfficientNetL2, MyEfficientNetL2_dropout



# [TODO: Optional] Rewrite this class if you want
class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str,
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork()
        elif model_name == 'MyAlexNet':
            self.model = MyAlexNet()
        elif model_name == 'MyConvNeXtV2':
            self.model = MyConvNeXtV2()
        elif model_name == 'MyEfficientNetL2':
            self.model = MyEfficientNetL2()
        elif model_name == 'MyEfficientNetL2_dropout':
            self.model = MyEfficientNetL2_dropout()
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.f1score = MyF1Score()
        self.accuracy = MyAccuracy()

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        # print('scores',scores.shape,scores)
        # print('y',y.shape,y)
        f1 = self.f1score(scores, y)
        accuracy = self.accuracy(scores, y)
        # self.log_dict({'loss/train': loss, 'accuracy/train': accuracy},
        #               on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({'loss/train': loss, 'f1score/tran': f1, 'accuracy/train': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        # print('scores',scores.shape,scores)
        # print('y',y.shape,y)
        f1 = self.f1score(scores, y)
        accuracy = self.accuracy(scores, y)
        # self.log_dict({'loss/val': loss, 'accuracy/val': accuracy},
        #               on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({'loss/val': loss, 'f1score/val': f1, 'accuracy/val': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
