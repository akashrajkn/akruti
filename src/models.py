import random
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class WordEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim=300, enc_hid_dim=256, dec_hid_dim=150, dropout=0.4, device=None):
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
        self.device      = device

        self.embedding   = nn.Embedding(input_dim, emb_dim)
        self.rnn         = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc_mu       = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_sigma    = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, src):
        # embedded         = self.dropout(self.embedding(src))
        embedded         = self.embedding(src)

        # print("--")
        # print(src.is_cuda)
        # print(src.size())
        # print(embedded.size())
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
        src      = src.permute(1, 0, 2)
        embedded = torch.stack([torch.mm(batch.float(), self.embedding)
                                for batch in src], dim=0)

        return embedded


class Attention(nn.Module):
    '''
    Attention over tag embeddings
    '''
    def __init__(self, tag_embed_dim=200, dec_hid_dim=512):
        '''
        tag_embed_dim -- tag embedding dimension
        dec_hid_dim   -- decoder hidden state dimension
        '''
        super(Attention, self).__init__()

        self.tag_embed_dim = tag_embed_dim
        self.dec_hid_dim   = dec_hid_dim

        self.attn  = nn.Linear(dec_hid_dim + tag_embed_dim, dec_hid_dim)
        self.v     = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, tag_embeds):
        '''
        hidden     -- hidden state of the decoder
        tag_embeds -- tag embeddings
        '''
        batch_size = tag_embeds.shape[0]
        src_len    = tag_embeds.shape[1]  # Should be seq_len: 11 in the case of Turkish

        # print('Hidden size:    {}'.format(str(hidden.size())))

        hidden     = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy     = torch.tanh(self.attn(torch.cat((hidden, tag_embeds), dim=2)))
        energy     = energy.permute(0, 2, 1)

        v          = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention  = torch.bmm(v, energy).squeeze(1)

        # print('Hidden size:    {}'.format(str(hidden.size())))
        # print('Tag Embed size: {}'.format(str(tag_embeds.size())))
        # print('v size:         {}'.format(str(v.size())))
        # print('self.v size:    {}'.format(str(self.v.size())))
        # print('energy size:    {}'.format(str(energy.size())))
        # print('attention size: {}'.format(str(attention.size())))

        return F.softmax(attention, dim=1)


class WordDecoder(nn.Module):
    def __init__(self, attention, output_dim, emb_dim=300, z_dim=150, tag_embed_dim=200, dec_hid_dim=512, dropout=0.4, device=None):
        super(WordDecoder, self).__init__()

        self.z_dim         = z_dim
        self.emb_dim       = emb_dim
        self.tag_embed_dim = tag_embed_dim
        self.dec_hid_dim   = dec_hid_dim
        self.output_dim    = output_dim  # vocab size
        self.attention     = attention
        self.device        = device

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn       = nn.GRU(emb_dim + tag_embed_dim + z_dim, dec_hid_dim)
        self.out       = nn.Linear(tag_embed_dim + dec_hid_dim, output_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, input, hidden, tag_embeds, z):
        '''
        input      -- previous input
        hidden     -- hidden state of the decoder
        tag_embeds -- tag embeddings
        z          -- lemma represented by the latent variable
        '''
        input = input.type(torch.LongTensor).to(self.device)
        input = input.unsqueeze(0)

        # embedded = self.dropout(self.embedding(input))
        embedded = self.embedding(input)

        a = self.attention(hidden, tag_embeds)
        a = a.unsqueeze(1)
        z = z.unsqueeze(0)  # For tensor dimension compatibility - see rnn_input

        # print(a.size())
        # print(z.size())
        # print(tag_embeds.size())

        weighted       = torch.bmm(a, tag_embeds)
        weighted       = weighted.permute(1, 0, 2)

        rnn_input      = torch.cat((embedded, weighted, z), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # TODO: Find out why this is required
        # assert (output == hidden).all()



        output   = output.squeeze(0)
        weighted = weighted.squeeze(0)
        output   = self.out(torch.cat((output, weighted), dim=1))

        # print("---")
        # print(output.size())
        # print(embedded.size())
        # print(weighted.size())
        # print(z.size())

        return output, hidden.squeeze(0)


class MSVED(nn.Module):
    '''
    Multi-space Variational Encoder-Decoder
    '''
    def __init__(self, encoder, tag_embedding, decoder, max_len, batch_size, vocab_size, device):
        super(MSVED, self).__init__()

        self.encoder       = encoder
        self.tag_embedding = tag_embedding
        self.decoder       = decoder

        self.max_len       = max_len
        self.device        = device
        self.batch_size    = batch_size
        self.vocab_size    = vocab_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x_s, x_t, y_t=None):
        # Semi supervised
        if y_t is None:
            pass

        outputs        = torch.zeros(self.max_len, self.batch_size, self.vocab_size).to(self.device)
        h, mu_u, var_u = self.encoder(x_s)
        tag_embeds     = self.tag_embedding(y_t)
        # z              = self.reparameterize(mu_u, var_u)

        o              = x_t[0, :]  # Start tokens

        for t in range(1, self.max_len):
            o, h       = self.decoder(o, h, tag_embeds, mu_u)
            outputs[t] = o
            o          = o.max(1)[1]

        return outputs, mu_u, var_u
