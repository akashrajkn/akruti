import random
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class VAE(nn.Module):
    '''
    Preliminaries: VAE
    '''
    def __init__(self, vocab_size, emb_dim, z_dim, h_dim, device, max_len, padding_idx=None):
        super(VAE, self).__init__()

        self.input_dim   = vocab_size
        self.output_dim  = vocab_size
        self.z_dim       = z_dim
        self.emb_dim     = emb_dim
        self.h_dim       = h_dim
        self.device      = device
        self.vocab_size  = vocab_size
        self.max_len     = max_len
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(self.input_dim, emb_dim, padding_idx=padding_idx)
        self.rnn_enc   = nn.GRU(emb_dim, h_dim, bidirectional=True)
        self.rnn_dec   = nn.GRU(emb_dim + z_dim, h_dim * 2)
        self.fc_mu     = nn.Linear(h_dim * 2, z_dim)
        self.fc_sigma  = nn.Linear(h_dim * 2, z_dim)
        self.out       = nn.Linear(h_dim * 2, self.output_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def encoder(self, x):
        embedded  = self.embedding(x)
        _, hidden = self.rnn_enc(embedded)
        hidden    = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        mu        = F.relu(self.fc_mu(hidden))
        sigma     = F.relu(self.fc_sigma(hidden))

        return hidden, mu, sigma

    def decoder(self, o, h, z):

        input    = o.type(torch.LongTensor).to(self.device)
        input    = input.unsqueeze(0)
        embedded = self.embedding(input)
        z        = z.unsqueeze(0)

        rnn_dec_input  = torch.cat((embedded, z), dim=2)
        output, hidden = self.rnn_dec(rnn_dec_input, h.unsqueeze(0))
        output         = output.squeeze(0)
        output         = self.out(output)

        return output, hidden.squeeze(0)

    def forward(self, x):

        batch_size       = x.shape[1]
        outputs          = torch.zeros(self.max_len, batch_size, self.vocab_size).to(self.device)

        h, z_mu, z_sigma = self.encoder(x)
        z                = self.reparameterize(z_mu, z_sigma)
        o                = x[0, :]

        for step in range(1, self.max_len):
            o, h          = self.decoder(o, h, z)
            outputs[step] = o
            o             = o.max(1)[1]

        return outputs, z_mu, z_sigma
