import sys
import os
import pickle
import logging
import argparse
import codecs
import time

import numpy    as np

from keras.utils import to_categorical
from msved       import *
from preprocess  import *
from utils       import *


def kl_div(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def prepare_sequence(sequence, char_2_idx, max_seq_len):
    '''
    Append <EOS> to each sequence and Pad with <PAD>
    '''
    output = ['SOS']

    for char in sequence:
        output.append(char_2_idx[char])

    output.append(char_2_idx['<EOS>'])

    while len(output) < max_seq_len:
        output.append(char_2_idx['<PAD>'])

    k_output = torch.from_numpy(to_categorical(output, num_classes=len(char_2_idx)))

    return k_output

def get_target(sequence, char_2_idx, max_seq_len):
    output = ['<SOS>']

    for char in sequence:
        output.append(char_2_idx[char])

    output.append(char_2_idx['<EOS>'])

    while len(output) < max_seq_len:
        output.append(char_2_idx['<PAD>'])

    return torch.tensor(output)

def prepare_msd(msd, idx_2_desc, msd_options):
    '''
    msd: {'pos': 'verb', 'tense': 'present', 'mod': 'ind'}

    output: [0, 1, 2, 0, 0, ...]
    '''
    label_len = len(idx_2_desc)
    k_output  = []

    for i in range(label_len):
        desc  = idx_2_desc[i]
        opt   = msd.get(desc)
        types = msd_options[i]

        if opt is None:
            k_output.append(to_categorical([0], num_classes=len(types))[0])
            continue

        k_output.append(to_categorical([types[opt]], num_classes=len(types))[0])

    return torch.from_numpy(np.concatenate(k_output, axis=0))

def main():
    # train_file    = '../data/files/task3_test'
    train_file    = '../data/files/turkish-task3-dev'

    idx_2_char    = load_file('../data/pickles/idx_2_char')
    char_2_idx    = load_file('../data/pickles/char_2_idx')
    idx_2_desc    = load_file('../data/pickles/idx_2_desc')
    desc_2_idx    = load_file('../data/pickles/desc_2_idx')
    msd_options   = load_file('../data/pickles/msd_options')

    epochs        = 20
    h_dim         = 256
    z_dim         = 150
    vocab_size    = len(char_2_idx)
    msd_size      = len(desc_2_idx)
    bidirectional = True

    training_data = read_task_3(train_file)
    max_seq_len   = max_sequence_length(train_file) + 2  # +2 is for <SOS> & <EOS> char
    label_len     = get_label_length(idx_2_desc, msd_options)

    model = MSVED(h_dim, z_dim, vocab_size, max_seq_len, label_len, bidirectional=bidirectional)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # TRAIN
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for triplet in training_data:
            model.zero_grad()

            x_s               = prepare_sequence(triplet['source_form'], char_2_idx, max_seq_len)
            y_t               = prepare_msd(triplet['MSD'], idx_2_desc, msd_options)
            x_t               = get_target(triplet['target_form'], char_2_idx, max_seq_len).type(torch.LongTensor)
            x_t_p, mu, logvar = model(x_s, y_t)

            loss = loss_function(x_t_p, x_t) + kl_div(mu, logvar)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.detach().cpu()

        print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss))

    # TEST
    test_file = '../data/files/turkish-task3-test'
    test_data = read_task_3(test_file)

    for triplet in test_data:

        with torch.no_grad():
            x_s = prepare_sequence(triplet['source_form'], char_2_idx, max_seq_len)
            y_t = prepare_msd(triplet['MSD'], idx_2_desc, msd_options)

            x_t_p, _, _ = model(x_s, y_t)

            outputs = F.log_softmax(x_t_p, dim=1)

            print('Target: {}')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    main()
