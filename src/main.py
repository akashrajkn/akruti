import sys
import os
import pickle
import logging
import argparse
import codecs
import time

import helper
import numpy     as np

from models      import *


def kl_div(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def prepare_sequence(sequence, char_2_idx, max_seq_len):
    '''
    Append <EOS> to each sequence and Pad with <PAD>
    '''
    output = [char_2_idx['<SOS>']]

    for char in sequence:
        output.append(char_2_idx[char])

    output.append(char_2_idx['<EOS>'])

    while len(output) < max_seq_len:
        output.append(char_2_idx['<PAD>'])

    print('Formatted Word: {}'.format(output))

    return torch.tensor(output).type(torch.LongTensor)


def prepare_msd(msd, idx_2_desc, msd_options):
    '''
    msd: {'pos': 'verb', 'tense': 'present', 'mod': 'ind'}

    output: [0, 5, 7, 10, ...] length: |label_types|
    '''
    label_types = len(idx_2_desc)
    output  = []

    for i in range(label_types):
        desc  = idx_2_desc[i]
        opt   = msd.get(desc)
        types = msd_options[i]

        if opt is None:
            opt = 'None'

        output.append(types[opt])

    print('Formatted MSD : {}'.format(output))

    return torch.tensor(output).type(torch.LongTensor)


def main():
    # train_file    = '../data/files/task3_test'
    train_file    = '../data/files/turkish-task3-dev'

    idx_2_char    = helper.load_file('../data/pickles/idx_2_char')
    char_2_idx    = helper.load_file('../data/pickles/char_2_idx')
    idx_2_desc    = helper.load_file('../data/pickles/idx_2_desc')
    desc_2_idx    = helper.load_file('../data/pickles/desc_2_idx')
    msd_options   = helper.load_file('../data/pickles/msd_options')  # label types

    epochs        = 20
    h_dim         = 256
    z_dim         = 150
    vocab_size    = len(char_2_idx)
    msd_size      = len(desc_2_idx)
    bidirectional = True

    training_data = helper.read_task_3(train_file)
    max_seq_len   = helper.max_sequence_length(train_file)
    label_len     = helper.get_label_length(idx_2_desc, msd_options)

    print(msd_options)
    print(label_len)

    print(idx_2_desc)


    device        = torch.device('cpu')
    encoder       = WordEncoder(vocab_size)   # TODO: give padding_idx
    tag_embedding = TagEmbedding(label_len)
    attention     = Attention()
    decoder       = Decoder(attention, vocab_size)

    model         = MSVED(encoder, tag_embedding, decoder, max_seq_len, device)

    print(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer     = optim.SGD(model.parameters(), lr=0.1)

    # TRAIN
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for triplet in training_data:
            model.zero_grad()

            x_s               = prepare_sequence(triplet['source_form'], char_2_idx, max_seq_len)
            y_t               = prepare_msd(triplet['MSD'], idx_2_desc, msd_options)
            x_t               = prepare_sequence(triplet['target_form'], char_2_idx, max_seq_len)

            x_s = torch.unsqueeze(x_s, 1)
            y_t = torch.unsqueeze(y_t, 1)
            x_t = torch.unsqueeze(x_t, 1)

            x_t_p, mu, logvar = model(x_s, y_t)

            loss = loss_function(x_t_p, x_t) + kl_div(mu, logvar)
            loss.backward()
            optimizer.step()

            break
        epoch_loss += loss.detach().cpu()
        print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss))
        break

    # TEST
    # test_file = '../data/files/turkish-task3-test'
    # test_data = read_task_3(test_file)
    #
    # for triplet in test_data:
    #
    #     with torch.no_grad():
    #         x_s = prepare_sequence(triplet['source_form'], char_2_idx, max_seq_len)
    #         y_t = prepare_msd(triplet['MSD'], idx_2_desc, msd_options)
    #
    #         x_t_p, _, _ = model(x_s, y_t)
    #
    #         outputs = F.log_softmax(x_t_p, dim=1)

            # print('Target: {}')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    main()
