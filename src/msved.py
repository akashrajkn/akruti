import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)


class MSVED(nn.Module):
    def __init__(self, h_dim, z_dim, vocab_size, max_len, label_len, bidirectional=True):
        super(MSVED, self).__init__()

        # u_dim = 2 * h_dim for bidirectional
        self.h_dim         = h_dim
        self.z_dim         = z_dim
        self.vocab_size    = vocab_size
        self.max_len       = max_len
        self.label_len     = label_len
        self.bidirectional = bidirectional

        # Encoder   --   TODO: Check if I can replace it with GRU layer
        self.rnn_forward  = nn.GRUCell(input_size=vocab_size, hidden_size=h_dim)
        self.rnn_backward = nn.GRUCell(input_size=vocab_size, hidden_size=h_dim)

        # TODO: initialize weight
        self.z_mu     = nn.Linear(2 * h_dim, z_dim)
        self.z_logvar = nn.Linear(2 * h_dim, z_dim)

        # Decoder
        # Supervised case
        self.rnn_decode   = nn.GRUCell(input_size=z_dim + label_len,
                                       hidden_size=h_dim)

        self.fc           = nn.Linear(h_dim, vocab_size)

    def encode(self, x_s):
        source_len = x_s.size()[0]
        h_forward  = torch.zeros(self.h_dim)
        h_forward  = torch.unsqueeze(h_forward, 0)
        h_backward = torch.zeros(self.h_dim)
        h_backward = torch.unsqueeze(h_backward, 0)

        for i in range(source_len):
            forward_char  = x_s[i]
            backward_char = x_s[source_len - i - 1]
            h_forward     = self.rnn_forward(torch.unsqueeze(forward_char, 0), h_forward)
            h_backward    = self.rnn_backward(torch.unsqueeze(backward_char, 0), h_backward)

        # u: hidden representation of x_s
        u      = torch.cat((h_forward, h_backward), 1)
        mu     = F.relu(self.z_mu(u))
        logvar = F.relu(self.z_logvar(u))

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z, y_t):
        target   = torch.empty((self.max_len, 1, self.vocab_size))
        h_decode = torch.zeros(self.h_dim)
        h_decode = torch.unsqueeze(h_decode, 0)

        for i in range(self.max_len):
            inp       = torch.cat((z, y_t), 1)
            h_decode  = self.rnn_decode(inp, h_decode)
            out       = self.fc(h_decode)

            target[i] = out

        return target

    def forward(self, x_s, y_t):

        # Add batch
        y_t        = torch.unsqueeze(y_t, 0)

        mu, logvar = self.encode(x_s)
        z          = self.reparameterize(mu, logvar)
        x_t        = self.decode(z, y_t)

        return x_t, mu, logvar
