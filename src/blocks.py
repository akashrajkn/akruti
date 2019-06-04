import random
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from kumaraswamy import Kumaraswamy
from hard_kumaraswamy import StretchedAndRectifiedDistribution as HardKumaraswamy

from models import WordEncoder, WordDecoder, TagEmbedding, Attention


class VAE(nn.Module):
    '''
    Preliminaries: VAE
    '''
    def __init__(self, input_dim, emb_dim, z_dim, h_dim, device, padding_idx=None):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.z_dim     = z_dim
        self.emb_dim     = emb_dim
        self.h_dim     = h_dim
        self.device    = device
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(input_dim, e_dim, padding_idx=padding_idx)
        self.rnn       = nn.GRU(emb_dim, h_dim, bidirectional=True)
        self.fc_mu      = nn.Linear(h_dim * 2, z_dim)
        self.fc_sigma   = nn.Linear(h_dim * 2, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def encoder(self, x):
        embedded  = self.embedding(x)
        _, hidden = self.rnn(embedded)
        hidden    = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        mu        = F.relu(self.fc_mu(hidden))
        sigma     = F.relu(self.fc_sigma(hidden))

        return hidden, mu, sigma

    def decoder(self):
        pass

    def forward(self, x):

        h, z_mu, z_sigma = self.encoder(x)
        z                = self.reparameterize(z_mu, z_sigma)
