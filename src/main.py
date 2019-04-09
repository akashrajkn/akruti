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
    output = []

    for char in sequence:
        output.append(char_2_idx[char])

    output.append(char_2_idx['<END>'])

    while len(output) < max_seq_len:
        output.append(char_2_idx['<PAD>'])

    return output

def prepare_msd(msd, idx_2_desc, msd_options):
    '''
    msd: {'pos': 'verb', 'tense': 'present', 'mod': 'ind'}

    output: [0, 1, 2, 0, 0, ...]
    '''
    label_len = len(idx_2_desc)
    output    = []

    for i in range(label_len):
        desc = idx_2_desc[i]
        opt  = msd.get(desc)

        if opt is None:
            output.append(0)
            continue

        types = msd_options[i]
        output.append(types[opt])

    return output

def main():
    train_file    = '../data/files/task3_test'

    epochs        = 20
    h_dim         = 256
    z_dim         = 150
    vocab_size    = len(char_2_idx)
    msd_size      = len(desc_2_idx)
    bidirectional = True

    idx_2_char    = load_file('../data/pickles/idx_2_char')
    char_2_idx    = load_file('../data/pickles/char_2_idx')
    idx_2_desc    = load_file('../data/pickles/idx_2_desc')
    desc_2_idx    = load_file('../data/pickles/desc_2_idx')
    msd_options   = load_file('../data/pickles/msd_options')

    training_data = load_file(train_file)
    max_seq_len   = max_sequence_length(train_file) + 1  # +1 is for <END> char
    label_len     = len(desc_2_idx)

    model = MSVED(h_dim, z_dim, vocab_size, msd_size, max_seq_len, label_len, bidirectional=bidirectional)
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # TRAIN

    model.train()
    for epoch in range(epochs):
        for triplet in training_data:
            model.zero_grad()

            x_s               = prepare_sequence(triplet['source_form'], char_2_idx, max_seq_len)
            y_t               = prepare_msd(triplet['MSD'], idx_2_desc, msd_options)
            x_t               = prepare_sequence(triplet['target_form'], char_2_idx, max_seq_len)
            x_t_p, mu, logvar = model(x_s, y_t)

            loss = loss_function(x_t_p, x_t) + kl_div(mu, logvar)

            loss.backward()
            optimizer.step()

    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    main()
