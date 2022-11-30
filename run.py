import os
import logging
import joblib
import json

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize 

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import hydra 
from hydra.utils import to_absolute_path as abs_path


from utils import losses
from utils.seed import set_seed
from utils.utils import *

from Model.model_resnet import *
from utils.dataset import WSIDataset
from batch_sampler import SequentialSampler


class LitAutoEncoder(pl.LightningModule):
    def __init__(
                self,
                cfg,
                fold,
                ):
        super().__init__()
        self.cfg = cfg
        self.fold = fold
        if cfg.model.resnet == 18:
            self.model = Resnet18(n_classes=3)
        elif cfg.model.resnet == 34:
            self.model = Resnet34(n_classes=3)
        else:
            self.model = Resnet50(n_classes=3)  
        self.loss_fn = losses.PropotionLoss()
        self.lr = cfg.train.lr
        # disable auto optim
        self.automatic_optimization = False
        self.labels = ['tumorbed', 'no_label', 'residual']
        self.resize = Resize([400, 400])
        tf = open(abs_path("propotion.json"), "r")
        self.propotion = json.load(tf)

    @torch.no_grad()
    def forward(self, ft, labels, batch_idx):
        # in lightning, forward defines the prediction/inference actions
        with torch.no_grad():
            for i in range(0, len(self.model.net_E)):
                output, ft = self.model(ft, layer_num=i)
        cm = confusion_score(labels, torch.argmax(output, dim=1)) 
        return {
                'cm': cm
               }   

    def _step(self, ft, propotion, layer_num=None):
        # in lightning, forward defines the prediction/inference actions
        output, ft = self.model(ft.to(self.device), layer_num)
        loss = self.loss_fn(output, propotion)
        return ft, loss, output

    def training_step(self, batch, batch_idx):
        ft, labels, name = batch['img'], batch['label'], batch['name']
        ft, labels = ft[0:500], labels[0:500]
        propotion = self.propotion[np.unique(name)[0]]
        for i in range(0, len(self.model.net_E)):
            ft, loss, output = self._step(ft, propotion, layer_num=i)
            self.manual_backward(loss / ft.shape[1])
            ft = ft.detach()
        if batch_idx // 4 == 0:
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()

        self.log('train_loss', 
                  loss.item(),
                  on_step=True,
                  prog_bar=True,
                  logger=True,
                )
        return loss

    def validation_step(self, batch, batch_idx):
        ft, labels, name = batch['img'], batch['label'], batch['name']
        ft, labels = ft[0:1000], labels[0:1000]
        propotion = self.propotion[np.unique(name)[0]]
        items = self.forward(ft, labels, batch_idx)    
        return items

    def test_step(self, batch, batch_idx):
        ft, labels, name = batch['img'], batch['label'], batch['name']
        ft, labels = ft[0:1000], labels[0:1000]
        # propotion = self.propotion[name]
        # print(np.unique(name))
        items = self.forward(ft, labels, batch_idx)
        return items

    def validation_epoch_end(self, items):
        cm_all = np.sum(np.stack([x['cm'] for x in items]), axis=0)
        metric_val = eval_metrics(cm_all)
        log_dict = {'validation_mIoU': metric_val['mIoU'], 'step': self.current_epoch}
        temp = metric_val['mIoU']
        logging.info(f'Validation mIoU: {temp}')
        self.log("val_loss", 
                  metric_val['mIoU'], 
                  logger=True,
                  on_epoch=True, 
                  sync_dist=True
                  )
        return {'log': log_dict, 
                'val_loss': metric_val['mIoU'], 
                'progress_bar': log_dict
                }

    def test_epoch_end(self, items):
        cm_all = np.sum(np.stack([x['cm'] for x in items]), axis=0)
        metric_val = eval_metrics(cm_all)
        np.set_printoptions(precision=2)
        plot_confusion_matrix(cm_all, classes=self.labels, normalize=True, title='Normalized confusion matrix')
        plt.savefig((abs_path(os.path.join(self.cfg.output.dir,  
                                           f'fold{self.fold + 1}', 
                                           f"confusion_matrix_fold{self.fold}.png")
                                           )), 
                                           bbox_inches = "tight")
        log_dict = {'test_miou': metric_val['mIoU'], 'step': self.current_epoch}  
        mIoU = metric_val['mIoU']
        logging.info(f'Test mIoU loss: {mIoU}')
        return {
                'log': log_dict, 
                'test_loss': mIoU, 
                'progress_bar': log_dict
                }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer


@hydra.main(config_path='config', config_name='config_lightning')
def main(cfg):
    set_seed(0)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda:{cfg.device.num}' if torch.cuda.is_available() else 'cpu')
    transform = {'Resize': False, 'HFlip': True, 'VFlip': True}

    for fold in range(cfg.train.fold):
        try:
            os.makedirs(abs_path(cfg.output.dir) + f'/fold{fold + 1}' + '/image')
            logging.info(f'Fold {fold + 1}/{cfg.train.fold}')
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        except OSError:
            pass

        train_wsis = joblib.load(abs_path('liu' + f'/cv{fold}_train_wsi.jb'))
        valid_wsis = joblib.load(abs_path('liu' + f'/cv{fold}_valid_wsi.jb'))
        test_wsis = joblib.load(abs_path('liu' + f'/cv{fold}_test_wsi.jb'))

        dataset = WSIDataset(
            imgs_dir=cfg.train.imgs,
            train_wsis=train_wsis,
            valid_wsis=valid_wsis,
            test_wsis=test_wsis,
            classes=[0, 1, 2],
            transform=transform
        )
        train_set, val_set, test_set = dataset.get()

        # train dataloader 
        train_sampler = SequentialSampler(imgs_dir=cfg.train.imgs, wsis=train_wsis, shuffle=True)
        train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=cfg.train.num_workers, shuffle=False)
        # val dataloader  
        val_sampler = SequentialSampler(imgs_dir=cfg.eval.imgs, wsis=valid_wsis, shuffle=False)            
        val_loader = DataLoader(val_set, batch_sampler=val_sampler, num_workers=cfg.eval.num_workers, shuffle=False)    
        # test dataloader 
        test_sampler = SequentialSampler(imgs_dir=cfg.eval.imgs, wsis=test_wsis, shuffle=False) 
        test_loader = DataLoader(test_set, batch_sampler=test_sampler, num_workers=cfg.eval.num_workers, shuffle=False)    

        logger = pl_loggers.TensorBoardLogger(abs_path(cfg.output.dir),
                                              name=f'fold{fold + 1}',
                                              version=f'Resnet{cfg.model.resnet}'
                                              )
        early_stop_callback = EarlyStopping(monitor='val_loss', 
                                            patience=cfg.train.early_stop, 
                                            check_finite=True,
                                            mode='max'
                                            )
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath = abs_path(cfg.output.dir) + f'/fold{fold + 1}' + f'/Resnet{cfg.model.resnet}/',
                                            #   filename='epoch{epoch:02d}-val_acc{val_loss:.2f}',
                                              filename='Best',
                                              mode='max'
                                              )

        pl.seed_everything(0)
        model = LitAutoEncoder(cfg, fold).to(device)
        trainer = pl.Trainer(
                            accelerator="gpu",    
                            gpus=[cfg.device.num],
                            check_val_every_n_epoch=1,
                            logger=logger,
                            max_epochs=cfg.train.epochs, 
                            callbacks=[checkpoint_callback, early_stop_callback],
                            num_sanity_val_steps=0
                            )
        trainer.fit(model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader
                    )
        trainer.test(model, 
                     dataloaders=test_loader,
                     ckpt_path= abs_path(cfg.output.dir) + f'/fold{fold + 1}/Resnet{cfg.model.resnet}/Best.ckpt'
                    )


if __name__ == '__main__':    
    main()