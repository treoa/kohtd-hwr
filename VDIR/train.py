import os
import torch
import hydra

import pytorch_lightning as pl
import torch.functional as F
import torch.nn as nn

from omegaconfig import OmegaConf
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import *


class vdir_train(pl.LightningModule):
    """
        VDIR Training Module
    """

    def __init__(self, hparams: dict, train_loader: DataLoader,
                 val_loader: DataLoader, test_loader: DataLoader,):
        """
            Initialize the VDIR training module
            Args:
                hparams: (dict)
        """
        super().__init__()
        self.hparams = hparams
        self.num_epochs = hparams['VDIR']['NUM_EPOCHS']
        self.learning_rate = self.hparams["VDIR"]["LR"]
        self.height, self.width, self.channels = hparams['VDIR']['INPUT_SHAPE']
        self.batch_size = self.hparams["VDIR"]["BATCH_SIZE"]
        self.step = self.hparams["VDIR"]["STEP"]
        self.model = self.hparams["VDIR"]["CHECKPOINT"] if self.hparams["VDIR"]["CHECKPOINT"] else None
        self.optimizer = None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.train_acc = None
        self.val_acc = None
        self.test_acc = None

        '''P(c|y) inference'''
        self.EST = Encoder(self.input, 'EST', feat=4)

        '''Re-parametrization trick'''
        eps = tf.random_normal(tf.shape(self.EST.mu))
        self.condition = eps*tf.exp(self.EST.sigma / 2.) + self.EST.mu

        '''P(x|y,c) inference'''
        self.MODEL = Denoiser(self.input, self.condition, 'Denoise')

        '''P(y|c) reconstruction'''
        self.DEC = Decoder(self.condition, 'DEC')

        '''DISCRIMINATOR'''
        self.DIS_real = Discriminator(self.input)
        self.DIS_fake = Discriminator(self.DEC.output, reuse=True)

    def custom_loss(self, y, y_pred):
        """
            Custom loss function
        """
        self.
