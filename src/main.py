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

    # print('Formatted Word: {}'.format(output))

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

    # print('Formatted MSD : {}'.format(output))

    return torch.tensor(output).type(torch.LongTensor)


def main():
    # train_file    = '../data/files/task3_test'
    train_file    = '../data/files/turkish-task3-dev'

    idx_2_char    = helper.load_file('../data/pickles/idx_2_char')
    char_2_idx    = helper.load_file('../data/pickles/char_2_idx')
    idx_2_desc    = helper.load_file('../data/pickles/idx_2_desc')
    desc_2_idx    = helper.load_file('../data/pickles/desc_2_idx')
    msd_options   = helper.load_file('../data/pickles/msd_options')  # label types

    epochs        = 5
    h_dim         = 256
    z_dim         = 150
    vocab_size    = len(char_2_idx)
    msd_size      = len(desc_2_idx)
    bidirectional = True

    training_data = helper.read_task_3(train_file)
    max_seq_len   = helper.max_sequence_length(train_file)
    label_len     = helper.get_label_length(idx_2_desc, msd_options)

    # print(msd_options)
    # print(label_len)
    # print(idx_2_desc)

    # device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device        = torch.device('cpu')

    print('device =', device)

    encoder       = WordEncoder(vocab_size)   # TODO: give padding_idx
    tag_embedding = TagEmbedding(label_len)
    attention     = Attention()
    decoder       = WordDecoder(attention, vocab_size)

    model         = MSVED(encoder, tag_embedding, decoder, max_seq_len, device).to(device)

    print('-' * 13)
    print('    MODEL')
    print('-' * 13)
    print(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer     = optim.SGD(model.parameters(), lr=0.1)

    print('-' * 13)
    print('    TRAIN')
    print('-' * 13)

    # TRAIN
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        count      = 0
        for triplet in training_data:
            optimizer.zero_grad()

            x_s = prepare_sequence(triplet['source_form'], char_2_idx, max_seq_len).to(device)
            y_t = prepare_msd(triplet['MSD'], idx_2_desc, msd_options).to(device)
            x_t = prepare_sequence(triplet['target_form'], char_2_idx, max_seq_len).to(device)

            x_s = torch.unsqueeze(x_s, 1)
            y_t = torch.unsqueeze(y_t, 1)
            x_t = torch.unsqueeze(x_t, 1)

            x_t_p, mu, logvar = model(x_s, y_t)

            x_t_p = x_t_p[1:].view(-1, x_t_p.shape[-1])
            x_t   = x_t[1:].view(-1)

            loss = loss_function(x_t_p, x_t) + kl_div(mu, logvar)
            loss.backward()
            optimizer.step()

            current_loss = loss.detach().cpu().item()
            epoch_loss += current_loss
            count += 1

        print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss / count))

    print('-' * 13)
    print('  SAVE MODEL')
    print('-' * 13)

    torch.save(model, '../models/model-epochs_{}.pt'.format(str(epochs)))
    # return
    print('-' * 13)
    print('    TEST')
    print('-' * 13)


    # TEST
    test_file = '../data/files/turkish-task3-test'
    test_data = helper.read_task_3(test_file)

    count = 0

    for triplet in test_data:
        with torch.no_grad():
            x_s = prepare_sequence(triplet['source_form'], char_2_idx, max_seq_len)
            y_t = prepare_msd(triplet['MSD'], idx_2_desc, msd_options)

            x_s = torch.unsqueeze(x_s, 1)
            y_t = torch.unsqueeze(y_t, 1)

            x_t_p, _, _ = model(x_s, y_t)

            x_t_p       = x_t_p[1:].view(-1, x_t_p.shape[-1])

            outputs     = F.log_softmax(x_t_p, dim=1).type(torch.LongTensor)
            outputs     = torch.squeeze(outputs, 1)
            target_word = ''

            for i in outputs:
                p = np.argmax(i, axis=0).detach().cpu().item()
                target_word += idx_2_char[p]

            print('Target   : {}'.format(triplet['target_form']))
            print('Predicted: {}'.format(target_word))

        count += 1
        if count == 1:
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    main()
