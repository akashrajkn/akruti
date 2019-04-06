import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)


class MSVED(nn.Module):
    def __init__(self, h_dim, z_dim, vocab_size, msd_size, bidirectional=True):
        super(MSVED, self).__init__()

        # u_dim = 2 * h_dim for bidirectional
        self.h_dim         = h_dim
        self.z_dim         = z_dim
        self.vocab_size    = vocab_size
        self.msd_size      = msd_size
        self.bidirectional = bidirectional

        # Encoder   --   TODO: Check if I can replace it with GRU layer
        self.rnn_forward  = nn.GRUCell(input_size=vocab_size, hidden_size=h_dim)
        self.rnn_backward = nn.GRUCell(input_size=vocab_size, hidden_size=h_dim)

        # TODO: initialize weight
        self.z_mu     = nn.Linear(2 * h_dim, z_dim)
        self.z_logvar = nn.Linear(2 * h_dim, z_dim)

        # Decoder   -- TODO

    def encode(x_s):
        source_len = x_s.size()[0]
        h_forward  = torch.zeros(self.hidden_size)
        h_backward = torch.zeros(self.hidden_size)

        for i in range(source_len):
            h_forward  = self.rnn_forward(x_s[i], h_forward)
            h_backward = self.rnn_backward(x_s[source_len - i - 1], h_backward)

        # u      : hidden representation of x_s
        u      = torch.cat((h_forward, h_backward), 0)
        mu     = F.relu(self.z_mu(u))
        logvar = F.relu(self.z_logvar(u))

        # Reparam: z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decoder(z, y_t):
        pass

    def forward(self, x_s, y_t):

        # z  : latent representation of lemma
        z   = self.encode(x_s)

        # x_t: Re-inflected form
        x_t = self.decode(z, y_t)

        return x_t
