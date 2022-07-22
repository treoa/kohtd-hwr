import os
import torch

import torch.nn as nn
import pytorch_lightning as pl

from tqdm import tqdm


class Encoder(pl.LightningModule):
    """
        Encoder
    """

    def __init__(self, input, name, feat=4):
        """
            Initialize the Encoder
            Args:
                input: (tensor)
                name: (str)
                feat: (int)
        """
        super().__init__()
        self.input = input
        self.name = name
        self.feat = feat
        self.mu = nn.Linear(self.feat, 1)
        self.sigma = nn.Linear(self.feat, 1)

    def forward(self, x):
        """
            Forward pass
            Args:
                x: (tensor)
            Returns:
                mu: (tensor)
                sigma: (tensor)
        """
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma

    def training_step(self, batch, batch_idx):
        """
            Training step
            Args:
                batch: (tensor)
                batch_idx: (int)
            Returns:
                loss: (tensor)
        """
        x, y = batch
        mu, sigma = self.forward(x)
        loss = -0.5 * (torch.log(sigma) + (y - mu)**2 / sigma**2).sum()
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """
            Validation step
            Args:
                batch: (tensor)
                batch_idx: (int)
            Returns:
                loss: (tensor)
        """
        x, y = batch
        mu, sigma = self.forward(x)
        loss = -0.5 * (torch.log(sigma) + (y - mu)**2 / sigma**2).sum()
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        """
            Validation epoch end
            Args:
                outputs: (list)
            Returns:
                val_loss: (tensor)
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}
