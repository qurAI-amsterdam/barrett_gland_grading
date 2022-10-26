import numpy as np
import os
import pandas as pd
import torch
from torch import nn
import argparse
import pytorch_lightning as pl
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import wandb
from utils import plot_confusion_matrix, plot_roc_curves
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SlideDataset(Dataset):
    def __init__(self, x, y, ids=None):
        self.slide_sequences = x
        self.slide_labels = y
        self.ids = ids

    def __len__(self):
        return len(self.slide_sequences)

    def get_slide_id(self, idx):
        return self.ids[idx]

    def __getitem__(self, idx):
        x = torch.Tensor(self.slide_sequences[idx])        # (L, H_in)
        y = torch.Tensor(self.slide_labels[idx])           # (1)
        return x, y


def accumulate_outputs(outputs):
    y_true = np.hstack([output['y_true'] for output in outputs])
    y_pred = np.hstack([output['y_pred'] for output in outputs])
    y_prob = np.concatenate([output['y_prob'] for output in outputs])
    return y_true, y_pred, y_prob


class SlideGradeModel(pl.LightningModule):
    """
    Base Pytorch Lighting Model for classification on slide level.
    """

    def __init__(self, exp_dir, run_dir, input_size=4, hidden_size=512, num_layers=1, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp_head = nn.Linear(hidden_size, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.exp_dir = exp_dir
        self.run_dir = run_dir

    def forward(self, x):
        """
        Args:
            x: a sequence of features derived from a patch with shape (B, S, D)

        Returns:
            y_hat: a batch of predictions (=logits) with shape (B, 1)

        """
        _, hidden = self.gru(x)
        output = self.mlp_head(hidden).squeeze()
        return output

    def configure_optimizers(self, init_lr=1e-4, wd=1e-5, scheduler_patience=50, scheduler_factor=0.2):
        optimizer = torch.optim.Adam(self.parameters(), lr=init_lr, weight_decay=wd)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience),
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val loss'
        }
        return [optimizer], [lr_scheduler]

    def test_epoch_end(self, outputs):
        y_true, y_pred, y_prob = accumulate_outputs(outputs)
        print('Evaluating on {} samples.'.format(len(y_true)))
        ids = np.load(os.path.join(self.exp_dir, 'y_test_names.npy'))

        # store results
        results_df = pd.DataFrame({'slide': ids, 'y_true': y_true, 'y_pred': y_pred,
                                   'p_nd': y_prob[:, 0], 'p_lgd': y_prob[:, 1], 'p_hgd': y_prob[:, 2]})
        print(results_df)
        results_save_path = os.path.join(self.run_dir, 'results.csv')
        print('Saving results to: {}'.format(results_save_path))
        results_df.to_csv(results_save_path, index=False)

        # compute metrics
        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        plot_confusion_matrix(cm, save_path=os.path.join(self.run_dir, 'test_cm.png'), pixel_level=False, kappa=kappa)
        plot_roc_curves(y_true, y_prob, save_path=self.run_dir, plot_roc_dysplasia=True)
        wandb.log({'confusion matrix': wandb.Image(os.path.join(self.run_dir, 'test_cm.png')),
                   'roc per class': wandb.Image(os.path.join(self.run_dir, 'test_roc_per_class.png')),
                   'roc dysplasia': wandb.Image(os.path.join(self.run_dir, 'test_roc_dysplasia.png'))})
        print('test acc: {:05.2f}, test kappa: {:.2f}, test auc {:.2f}'.format(acc, kappa, auc))
        self.log_dict({'test acc': acc, 'test kappa': kappa, 'test auc': auc})

    def validation_epoch_end(self, outputs):
        y_true, y_pred, y_prob = accumulate_outputs(outputs)
        print('Evaluating on {} samples.'.format(len(y_true)))
        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        print('val acc: {:05.2f}, val kappa: {:.2f}, val auc {:.2f}'.format(acc, kappa, auc))
        self.log_dict({'val acc': acc, 'val kappa': kappa, 'val auc': auc})

    def training_step(self, train_batch, batch_idx):
        # forward and predict
        x, y = train_batch
        y_logits = self.forward(x)
        loss = self.loss(y_logits, y.long().squeeze())

        # log loss
        self.log_dict({'train loss': loss.item()})
        return loss

    def validation_step(self, val_batch, batch_idx):
        # forward and predict
        x, y = val_batch
        y_logits = self.forward(x)
        y_prob = torch.softmax(y_logits, dim=1)
        loss = self.loss(y_logits, y.long().squeeze())

        # to numpy
        y_true = y.long().detach().cpu().numpy().flatten()
        y_prob = y_prob.detach().cpu().numpy()
        y_pred = np.argmax(y_prob, axis=1)

        # log loss and auc
        self.log_dict({'val loss': loss.item()})
        return {'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob}

    def test_step(self, test_batch, batch_idx):
        # forward and predict
        x, y = test_batch
        y_logits = self.forward(x)
        y_prob = torch.softmax(y_logits, dim=1)

        # to numpy
        y_true = y.long().detach().cpu().numpy().flatten()
        y_prob = y_prob.detach().cpu().numpy()
        y_pred = np.argmax(y_prob, axis=1)
        return {'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob}


def train(run_name, nr_epochs, batch_size, lr, wd, experiments_dir, wandb_key, test_mode):
    """ Train something (Transformer/LSTM/GRU/GraphNN) for slide level classification.
    """
    seed_everything(5, workers=True)

    # load data
    x_train = np.load(os.path.join(experiments_dir, 'x_train_overlap.npy'))
    y_train = np.load(os.path.join(experiments_dir, 'y_train_overlap.npy'))
    x_val = np.load(os.path.join(experiments_dir, 'x_val_overlap.npy'))
    y_val = np.load(os.path.join(experiments_dir, 'y_val_overlap.npy'))

    # make train loaders
    train_dataset = SlideDataset(x_train, y_train)
    val_dataset = SlideDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=28, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=28)

    # create model & configure optimizer
    hidden_size = 512
    num_layers = 1
    model = SlideGradeModel(exp_dir=experiments_dir,
                            run_dir=os.path.join(experiments_dir, run_name),
                            input_size=4,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            num_classes=3)
    model.configure_optimizers(init_lr=lr, wd=wd)

    # log everything
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb_logger = WandbLogger(project="Barrett's Slide Classification", name=run_name, save_dir=experiments_dir)
    wandb_logger.log_hyperparams({
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": nr_epochs,
        "weight_decay": wd,
        "hidden size": hidden_size,
        "num layers": num_layers
    })
    wandb.run.name = run_name

    # define checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(experiments_dir, run_name),
                                          filename='{epoch:02d}_{step:03d}_{val loss:.2f}',
                                          monitor='val loss', mode='min', save_top_k=5)

    # define trainer
    trainer = Trainer(logger=wandb_logger,
                      callbacks=checkpoint_callback,
                      accelerator='gpu',
                      devices=1,
                      strategy='dp',
                      max_epochs=nr_epochs,
                      log_every_n_steps=1,
                      deterministic=True)

    # train the model
    trainer.fit(model, train_loader, val_loader)

    # use the best model just trained
    if test_mode:

        # make train loaders
        x_test = np.load(os.path.join(experiments_dir, 'x_test_overlap.npy'))
        y_test = np.load(os.path.join(experiments_dir, 'y_test_overlap.npy'))
        test_dataset = SlideDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=28, drop_last=False, shuffle=False)
        print('Testing model: {}'.format(trainer.checkpoint_callback.best_model_path))

        # get results on the test set
        trainer.test(dataloaders=test_loader, ckpt_path='best', verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='test', help="the name of this experiment")
    parser.add_argument("--nr_epochs", type=int, default=250, help="the number of epochs")
    parser.add_argument("--batch_size", type=int, default=4096*2, help="the size of mini batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="initial the learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay (L2)")
    parser.add_argument("--exp_dir", type=str,
                        default='/data/archief/AMC-data/Barrett/experiments/barrett_slide_classification/top_25_entropy',
                        help="experiment dir classification")
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")
    parser.add_argument("--test", type=bool, help="whether to also test", default=True)
    args = parser.parse_args()
    train(args.run_name, args.nr_epochs, args.batch_size, args.lr, args.wd, args.exp_dir, args.wandb_key, args.test)
