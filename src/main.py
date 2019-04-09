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

def prepare_sequence(sequence, char_2_idx, max_seq_len):
    '''
    Append <END> to each sequence and Pad with <PAD>
    '''
    pass

def prepare_msd(msd, desc_2_idx):
    pass

def main():

    train_file    = '../data/files/task3_test'

    epochs        = 20
    h_dim         = 200
    z_dim         = 300
    vocab_size    = len(char_2_idx)
    msd_size      = len(desc_2_idx)
    bidirectional = True

    idx_2_char    = load_file('../data/pickles/idx_2_char')
    char_2_idx    = load_file('../data/pickles/char_2_idx')
    idx_2_desc    = load_file('../data/pickles/idx_2_desc')
    desc_2_idx    = load_file('../data/pickles/desc_2_idx')

    training_data = load_file(train_file)
    max_seq_len   = max_sequence_length(train_file)

    model = MSVED(h_dim, z_dim, vocab_size, msd_size, bidirectional=bidirectional)
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        for triplet in training_data:
            model.zero_grad()

            x_s               = prepare_sequence(triplet['source_form'], char_2_idx, max_seq_len)
            y_t               = prepare_msd(triplet['MSD'], desc_2_idx)
            x_t               = prepare_sequence(triplet['target_form'], char_2_idx, max_seq_len)
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
