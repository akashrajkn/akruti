import random
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from embedding_mul import EmbeddingMul


class WordEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim=300, enc_hid_dim=256, dec_hid_dim=150, dropout=0.4):
        '''
        input_dim   -- vocabulary(characters) size
        emb_dim     -- character embedding dimension
        enc_hid_dim -- RNN hidden state dimenion
        dec_hid_dim -- latent variable z dimension
        '''

        super(WordEncoder, self).__init__()

        self.input_dim   = input_dim
        self.emb_dim     = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        # TODO: replace with EmbeddingMul
        self.embedding   = nn.Embedding(input_dim, emb_dim)
        self.rnn         = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc_mu       = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_sigma    = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, src):
        # embedded         = self.dropout(self.embedding(src))
        embedded         = self.embedding(src)
        _, hidden        = self.rnn(embedded)
        hidden           = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        z_mu             = torch.tanh(self.fc_mu(hidden))
        z_logvar         = torch.tanh(self.fc_sigma(hidden))

        return z_mu, z_logvar


class TagEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim=200):
        '''
        input_dim -- vocabulary(tags) size
        emb_dim   -- character embedding dimension
        '''
        super(TagEmbedding, self).__init__()

        self.input_dim = input_dim
        self.emb_dim   = emb_dim

        # TODO: replace with EmbeddingMul
        self.embedding = nn.Embedding(input_dim, emb_dim)

    def forward(self, src):
        embedded       = self.embedding(src)

        return embedded


class Attention(nn.Module):
    '''
    Attention over tag embeddings
    '''
    def __init__(self, tag_embed_dim=200, dec_hid_dim=256):
        '''
        tag_embed_dim -- tag embedding dimension
        dec_hid_dim   -- decoder hidden state dimension
        '''
        super(Attention, self).__init__()

        self.tag_embed_dim = tag_embed_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn  = nn.Linear(dec_hid_dim + tag_embed_dim, dec_hid_dim)
        self.v     = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, tag_embeds):

        batch_size = tag_embeds.shape[1]
        src_len    = tag_embeds.shape[0]

        hidden     = hidden.unsqueeze(1).repeat(1, src_len, 1)
        tag_embeds = tag_embeds.permute(1, 0, 2)

        energy     = torch.tanh(self.attn(torch.cat((hidden, tag_embeds), dim = 2)))
        energy     = energy.permute(0, 2, 1)

        v          = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention  = torch.bmm(v, energy).squeeze(1)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, attention, output_dim, z_dim=150, tag_embed_dim=200, dec_hid_dim=256, dropout=0.4):
        super(Decoder, self).__init__()

        self.z_dim         = z_dim
        self.tag_embed_dim = tag_embed_dim
        self.dec_hid_dim   = dec_hid_dim
        self.output_dim    = output_dim
        self.attention     = attention

        self.rnn     = nn.GRU(tag_embed_dim + z_dim, dec_hid_dim)
        self.out     = nn.Linear(tag_embed_dim + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, tag_embeds, z):
        '''
        hidden     -- hidden state of the decoder
        tag_embeds -- tag embeddings
        z          -- lemma represented by the latent variable
        '''
        a        = self.attention(hidden, tag_embeds)
        a        = a.unsqueeze(1)

        tag_embeds      = tag_embeds.permute(1, 0, 2)
        weighted        = torch.bmm(a, tag_embeds)
        weighted        = weighted.permute(1, 0, 2)
        rnn_input       = torch.cat((weighted, z), dim = 2)
        output, hidden  = self.rnn(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()

        output   = output.squeeze(0)
        weighted = weighted.squeeze(0)
        output   = self.out(torch.cat((output, weighted), dim = 1))

        return output, hidden.squeeze(0)


class MSVED(nn.Module):
    '''
    Multi-space Variational Encoder-Decoder
    '''
    def __init__(self, encoder, tag_embedding, decoder, max_len, device):
        super(MSVED, self).__init__()

        self.encoder       = encoder
        self.tag_embedding = tag_embedding
        self.decoder       = decoder

        self.max_len       = max_len
        self.device        = device

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x_s, y_t):
        batch_size     = x_s.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # encoder_outputs, hidden = self.encoder(x_s)
        z_mu, z_logvar = self.encoder(x_s)
        z              = self.reparameterize(z_mu, z_logvar)
        tag_embeds     = self.tag_embedding(y_t)

        outputs        = torch.zeros(self.max_len, batch_size, trg_vocab_size).to(self.device)
        h              = torch.zeros_like(outputs)

        for t in range(1, self.max_len):
            o, h       = self.decoder(h, tag_embeds, z)  # TODO: replace with z_mu
            outputs[t] = o

        return outputs, z_mu, z_logvar
