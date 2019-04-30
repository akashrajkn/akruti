import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from timeit import default_timer as timer
from datetime import timedelta

from models import WordEncoder, Attention, TagEmbedding, WordDecoder, MSVED
from helper import load_file
from dataset import MorphologyDatasetTask3


def kl_div(mu, logvar):
    '''
    Compute KL divergence between N(mu, logvar) and N(0, 1)
    '''
    return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def test(model, test_dataloader, config, idx_2_char, guesses_file):
    '''
    Test function
    TODO implement accuracy
    '''
    print('-' * 13)
    print('    TEST')
    print('-' * 13)

    device = config['device']
    output = ''

    model.eval()
    for i_batch, sample_batched in enumerate(test_dataloader):

        with torch.no_grad():
            x_s = sample_batched['source_form'].to(device)
            y_t = sample_batched['msd'].to(device)

            x_s = torch.transpose(x_s, 0, 1)
            y_t = torch.transpose(y_t, 0, 1)

            x_t_p, _, _ = model(x_s, y_t)
            x_t_p       = x_t_p[1:].view(-1, x_t_p.shape[-1])

            outputs     = F.log_softmax(x_t_p, dim=1).type(torch.LongTensor)
            outputs     = torch.squeeze(outputs, 1)

            target_word = ''
            for i in outputs:
                p      = np.argmax(i, axis=0).detach().cpu().item()
                entity = idx_2_char[p]

                if   entity == '<SOS>':
                    continue
                elif entity == '<PAD>' or entity == '<EOS>':
                    break

                target_word += idx_2_char[p]

            output += '{}\t{}\t{}\n'.format(sample_batched['source_str'][0], sample_batched['msd_str'][0], target_word)

    with open('../results/{}'.format(guesses_file), 'w+', encoding="utf-8") as f:
        f.write(output)


def train(train_dataloader, config, model_file):
    '''
    Train function
    '''
    device        = config['device']
    # Model declaration
    encoder       = WordEncoder(config['vocab_size'], device=device)  # TODO: give padding_idx
    tag_embedding = TagEmbedding(config['label_len'], device=device)
    attention     = Attention()
    decoder       = WordDecoder(attention, config['vocab_size'])
    model         = MSVED(encoder, tag_embedding, decoder, config['max_seq_len'], device).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer     = optim.SGD(model.parameters(), lr=0.1)

    kl_weight     = config['kl_start']
    anneal_rate   = (1.0 - config['kl_start']) / (config['epochs'] * len(train_dataloader))

    print('-' * 13)
    print('    DEVICE')
    print('-' * 13)
    print(device)
    print('-' * 13)
    print('    MODEL')
    print('-' * 13)
    print(model)
    print('-' * 13)
    print('    TRAIN')
    print('-' * 13)

    model.train()
    for epoch in range(config['epochs']):

        start      = timer()
        epoch_loss = 0

        for i_batch, sample_batched in enumerate(train_dataloader):
            optimizer.zero_grad()

            x_s = sample_batched['source_form'].to(device)
            y_t = sample_batched['msd'].to(device)
            x_t = sample_batched['target_form'].to(device)

            x_s = torch.transpose(x_s, 0, 1)
            y_t = torch.transpose(y_t, 0, 1)
            x_t = torch.transpose(x_t, 0, 1)

            x_t_p, mu, logvar = model(x_s, y_t)

            x_t_p = x_t_p[1:].view(-1, x_t_p.shape[-1])
            x_t   = x_t[1:].contiguous().view(-1)

            loss  = loss_function(x_t_p, x_t) + kl_weight * kl_div(mu, logvar)
            loss.backward()
            optimizer.step()

            kl_weight    = min(config['lambda_m'], kl_weight + anneal_rate)
            epoch_loss  += loss.detach().cpu().item()

        end = timer()

        print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss / i_batch))
        print('         - Time: {}'.format(timedelta(seconds=end - start)))

        print('         - Save model')
        torch.save(model, '../models/model-{}-epochs_{}.pt'.format(model_file, str(config['epochs'])))

    return model


if __name__ == "__main__":
    # Set up data
    idx_2_char = load_file('../data/pickles/idx_2_char')
    char_2_idx = load_file('../data/pickles/char_2_idx')
    idx_2_desc = load_file('../data/pickles/idx_2_desc')
    desc_2_idx = load_file('../data/pickles/desc_2_idx')
    msd_types  = load_file('../data/pickles/msd_options')  # label types

    parser     = argparse.ArgumentParser()
    parser.add_argument('--train',     action='store_true')
    parser.add_argument('--test',      action='store_true')
    parser.add_argument('-epochs',     action="store", type=int,   default=50)
    parser.add_argument('-h_dim',      action="store", type=int,   default=256)
    parser.add_argument('-z_dim',      action="store", type=int,   default=150)
    parser.add_argument('-batch_size', action="store", type=int,   default=64)
    parser.add_argument('-kl_start',   action="store", type=float, default=0.0)
    parser.add_argument('-lambda_m',   action="store", type=float, default=0.2)

    args       = parser.parse_args()
    run_train  = args.train
    run_test   = args.test

    config                  = {}
    config['epochs']        = args.epochs
    config['kl_start']      = args.kl_start
    config['h_dim']         = args.h_dim
    config['z_dim']         = args.z_dim
    config['lambda_m']      = args.lambda_m
    config['batch_size']    = args.batch_size
    config['language']      = 'turkish'
    config['device']        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['vocab_size']    = len(char_2_idx)

    # Get train_dataloader
    train_file       = '{}-task3-train'.format(config['language'])
    morph_data       = MorphologyDatasetTask3(csv_file='../data/files/{}.csv'.format(train_file), language=config['language'])
    morph_data.set_vocabulary(char_2_idx, idx_2_char, desc_2_idx, idx_2_desc, msd_types)
    train_dataloader = DataLoader(morph_data, batch_size=config['batch_size'], shuffle=True, num_workers=2, drop_last=True)

    config['max_seq_len']   = morph_data.max_seq_len
    config['label_len']     = morph_data.label_len

    # TRAIN
    if run_train:
        model = train(train_dataloader, config, model_file=train_file)

    # TEST
    if run_test:
        if not run_train:
            model = torch.load('../models/model-{}-epochs_{}.pt'.format(train_file, str(config['epochs'])))

        # Get test_dataloader
        test_file             = '{}-task3-test'.format(config['language'])
        test_morph_data       = MorphologyDatasetTask3(csv_file='../data/files/{}.csv'.format(test_file), language=config['language'], get_unprocessed=True)
        test_morph_data.set_vocabulary(char_2_idx, idx_2_char, desc_2_idx, idx_2_desc, msd_types)
        test_dataloader       = DataLoader(test_morph_data, batch_size=1, shuffle=False, num_workers=2)

        test(model, test_dataloader, config, idx_2_char, '{}-task3-guesses'.format(config['language']))
