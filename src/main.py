import sys
import os
import pickle
import logging
import argparse
import codecs
import time

import numpy    as np

from msved      import *
from preprocess import *
from utils      import *


def kl_div(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def main():

    idx_2_char = load_file('../data/pickles/idx_2_char')
    char_2_idx = load_file('../data/pickles/char_2_idx')
    idx_2_desc = load_file('../data/pickles/idx_2_desc')
    desc_2_idx = load_file('../data/pickles/desc_2_idx')

    epochs        = 20
    h_dim         = 200
    z_dim         = 300
    vocab_size    = len(char_2_idx)
    msd_size      = len(desc_2_idx)
    bidirectional = True

    model = MSVED(h_dim, z_dim, vocab_size, msd_size, bidirectional=bidirectional)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        for sentence, tags in training_data:
            model.zero_grad()

            x_s               = prepare_sequence(sentence, word_to_ix)
            x_t               = prepare_sequence(tags, tag_to_ix)
            x_t_p, mu, logvar = model(x_s, y_t)

            loss = loss_function(predicted, x_t) + kl_div(mu, logvar)

            loss.backward()
            optimizer.step()

    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    main()
