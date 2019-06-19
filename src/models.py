import random
import math
import time

import numpy as np

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from kumaraswamy import Kumaraswamy
from hard_kumaraswamy import StretchedAndRectifiedDistribution as HardKumaraswamy


class WordEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim=300, enc_h_dim=256, z_dim=150,
                 dropout=0.0, padding_idx=None, device=None):
        '''
        input_dim   -- vocabulary(characters) size
        emb_dim     -- character embedding dimension
        enc_h_dim -- RNN hidden state dimenion
        z_dim       -- latent variable z dimension
        '''
        super(WordEncoder, self).__init__()

        self.input_dim   = input_dim
        self.emb_dim     = emb_dim
        self.enc_h_dim   = enc_h_dim
        self.z_dim       = z_dim
        self.device      = device
        self.padding_idx = padding_idx

        self.embedding   = nn.Embedding(input_dim, emb_dim, padding_idx=padding_idx)
        self.rnn         = nn.GRU(emb_dim, enc_h_dim, bidirectional=True)
        self.fc_mu       = nn.Linear(enc_h_dim * 2, z_dim)
        self.fc_sigma    = nn.Linear(enc_h_dim * 2, z_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, src):
        embedded         = self.dropout(self.embedding(src))
        _, hidden        = self.rnn(embedded)
        hidden           = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        z_mu             = F.relu(self.fc_mu(hidden))
        z_logvar         = F.relu(self.fc_sigma(hidden))

        return hidden, z_mu, z_logvar


class TagEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim=200, device=None):
        '''
        input_dim  -- label length
        emb_dim    -- character embedding dimension
        device     -- cuda or cpu
        '''
        super(TagEmbedding, self).__init__()

        self.input_dim = input_dim
        self.emb_dim   = emb_dim
        self.device    = device

        self.embedding = nn.Parameter(torch.rand(input_dim, emb_dim))

    def forward(self, src):
        src            = src.permute(1, 0, 2)
        embedded       = torch.stack([torch.mm(batch.float(), self.embedding)
                                      for batch in src], dim=0)

        return embedded


class Attention(nn.Module):
    '''
    Attention over tag embeddings
    '''
    def __init__(self, emb_dim=200, dec_h_dim=512):
        '''
        emb_dim   -- tag embedding dimension
        dec_h_dim -- decoder hidden state dimension
        '''
        super(Attention, self).__init__()

        self.emb_dim   = emb_dim
        self.dec_h_dim = dec_h_dim

        self.attn      = nn.Linear(dec_h_dim + emb_dim, dec_h_dim)
        self.v         = nn.Parameter(torch.rand(dec_h_dim))

    def forward(self, hidden, tag_embeds):
        '''
        hidden     -- hidden state of the decoder
        tag_embeds -- tag embeddings
        '''
        batch_size = tag_embeds.shape[0]
        src_len    = tag_embeds.shape[1]  # sequence length

        hidden     = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy     = torch.tanh(self.attn(torch.cat((hidden, tag_embeds), dim=2)))
        energy     = energy.permute(0, 2, 1)

        v          = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention  = torch.bmm(v, energy).squeeze(1)

        return F.softmax(attention, dim=1)


class WordDecoder(nn.Module):
    def __init__(self, attention, output_dim, char_emb_dim=300, z_dim=150, tag_emb_dim=200,
                 dec_h_dim=512, dropout=0.4, padding_idx=None, device=None, no_attn=False):
        '''
        output_dim -- vocabulary size
        '''
        super(WordDecoder, self).__init__()

        self.z_dim        = z_dim
        self.char_emb_dim = char_emb_dim
        self.tag_emb_dim  = tag_emb_dim
        self.dec_h_dim    = dec_h_dim
        self.output_dim   = output_dim
        self.attention    = attention
        self.device       = device
        self.no_attn      = no_attn

        self.embedding    = nn.Embedding(output_dim, char_emb_dim)
        self.rnn          = nn.GRU(char_emb_dim + tag_emb_dim + z_dim, dec_h_dim)
        self.out          = nn.Linear(tag_emb_dim + dec_h_dim, output_dim)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, input, hidden, tag_embeds, z, drop):
        '''
        input      -- previous input
        hidden     -- hidden state of the decoder
        tag_embeds -- tag embeddings
        z          -- lemma represented by the latent variable
        '''
        input    = input.type(torch.LongTensor).to(self.device)
        input    = input.unsqueeze(0)
        # embedded = self.dropout(self.embedding(input))
        embedded = self.embedding(input)
        z        = z.unsqueeze(0)  # For tensor dimension compatibility - see rnn_input

        if drop:
            embedded = embedded * 0.

        # print("=-----")
        # print(embedded)
        # print(embedded.size())

        if self.no_attn:
            a = torch.ones((tag_embeds.size(0), 1, tag_embeds.size(1))).to(self.device)
        else:
            a = self.attention(hidden, tag_embeds)
            a = a.unsqueeze(1)

        weighted       = torch.bmm(a, tag_embeds)
        weighted       = weighted.permute(1, 0, 2)

        rnn_input      = torch.cat((embedded, weighted, z), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # TODO: Find out why this is required
        # assert (output == hidden).all()

        output   = output.squeeze(0)
        weighted = weighted.squeeze(0)
        output   = self.out(torch.cat((output, weighted), dim=1))

        return output, hidden.squeeze(0)


class MSVED(nn.Module):
    '''
    Multi-space Variational Encoder-Decoder
    '''
    def __init__(self, encoder, tag_embedding, decoder, max_len, vocab_size, device):
        super(MSVED, self).__init__()

        self.encoder       = encoder
        self.tag_embedding = tag_embedding
        self.decoder       = decoder

        self.max_len       = max_len
        self.device        = device
        self.vocab_size    = vocab_size

        # dropout: 40%
        self.dropout_dist   = dist.bernoulli.Bernoulli(torch.tensor([0.4]))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x_s, x_t, y_t):

        batch_size     = x_s.shape[1]
        outputs        = torch.zeros(self.max_len, batch_size, self.vocab_size).to(self.device)

        h, mu_u, var_u = self.encoder(x_s)
        tag_embeds     = self.tag_embedding(y_t)
        # z              = self.reparameterize(mu_u, var_u)
        o              = x_t[0, :]  # Start tokens

        for t in range(1, self.max_len):
            drop       = (self.dropout_dist.sample() == torch.tensor([1.]))
            o, h       = self.decoder(o, h, tag_embeds, mu_u, drop)
            outputs[t] = o
            o          = o.max(1)[1]

        return outputs, mu_u, var_u


class KumaMSD(nn.Module):
    '''
    Generates samples of y_t (MSD) vector.
    '''
    def __init__(self, input_dim, h_dim, num_tags, encoder, device, l=-1., r=2., unconstrained=False, use_made=False):
        super(KumaMSD, self).__init__()

        self.input_dim     = input_dim
        self.h_dim         = h_dim
        self.num_tags      = num_tags
        self.encoder       = encoder
        self.device        = device
        self.support       = [-l, r]
        self.unconstrained = unconstrained
        self.use_made      = use_made

        # Learned kuma params
        self.a  = 0.
        self.b  = 0.

        # TODO: initialization
        self.fc = nn.Linear(input_dim, h_dim)
        self.ai = nn.Linear(h_dim, num_tags)
        self.bi = nn.Linear(h_dim, num_tags)

    def forward(self, x_t):

        h, _, _ = self.encoder(x_t)
        logits  = F.relu(self.fc(h))

        if self.use_made:
            u_tril = torch.ones(self.h_dim, self.h_dim)
            u_tril = torch.tensor(np.tril(u_tril), requies_grad=False).to(self.device)
            logits = torch.mm(logits, u_tril)

        if self.unconstrained:
            ai   = F.softplus(self.ai(logits))
            bi   = F.softplus(self.bi(logits))
            kuma = Kumaraswamy(ai, bi)
        else:
            ai   = 0.1 + torch.sigmoid(self.ai(logits))     * 0.8
            bi   = 1   + torch.sigmoid(self.bi(logits) - 5) * 5
            kuma = Kumaraswamy(ai / bi, (1 - ai) / bi)

        self.a = ai
        self.b = bi

        h_kuma = HardKumaraswamy(kuma)
        sample = h_kuma.rsample()

        return sample, h_kuma
