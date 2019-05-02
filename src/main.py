import os
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


def initialize_model(config):
    '''
    Initialize and return the models
    Args:
        config -- config dict
    Return:
        model
    '''
    if not torch.cuda.is_available():
        device    = torch.device('cpu')
    else:
        device    = torch.device(config['device'])

    encoder       = WordEncoder(input_dim   =config['vocab_size'],
                                emb_dim     =config['char_emb_dim'],
                                enc_h_dim   =config['enc_h_dim'],
                                z_dim       =config['z_dim'],
                                dropout     =config['enc_dropout'],
                                padding_idx =config['padding_idx'],
                                device      =device)

    tag_embedding = TagEmbedding(input_dim  =config['label_len'],
                                 emb_dim    =config['tag_emb_dim'],
                                 device     =device)

    attention     = Attention(emb_dim       =config['tag_emb_dim'],
                              dec_h_dim     =config['dec_h_dim'])

    decoder       = WordDecoder(attention   =attention,
                                output_dim  =config['vocab_size'],
                                char_emb_dim=config['char_emb_dim'],
                                z_dim       =config['z_dim'],
                                tag_emb_dim =config['tag_emb_dim'],
                                dec_h_dim   =config['dec_h_dim'],
                                dropout     =config['dec_dropout'],
                                padding_idx =config['padding_idx'],
                                device      =device)

    model         = MSVED(encoder           =encoder,
                          tag_embedding     =tag_embedding,
                          decoder           =decoder,
                          max_len           =config['max_seq_len'],
                          vocab_size        =config['vocab_size'],
                          device            =device).to(device)

    return model


def initialize_dataloader(run_type, language, batch_size, shuffle):
    '''
    Initializes train and test dataloaders
    '''
    # Set up data
    idx_2_char = load_file('../data/pickles/{}-idx_2_char'.format(language))
    char_2_idx = load_file('../data/pickles/{}-char_2_idx'.format(language))
    idx_2_desc = load_file('../data/pickles/{}-idx_2_desc'.format(language))
    desc_2_idx = load_file('../data/pickles/{}-desc_2_idx'.format(language))
    msd_types  = load_file('../data/pickles/{}-msd_options'.format(language))  # label types

    file_name  = '{}-task3-{}'.format(language, run_type)
    morph_data = MorphologyDatasetTask3(csv_file='../data/files/{}.csv'.format(file_name),
                                                   language=language, get_unprocessed=(run_type == 'test'))

    morph_data.set_vocabulary(char_2_idx, idx_2_char, desc_2_idx, idx_2_desc, msd_types)
    dataloader = DataLoader(morph_data, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return dataloader, morph_data


def kl_div(mu, logvar):
    '''
    Compute KL divergence between N(mu, logvar) and N(0, 1)
    '''
    return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def test(language, model_id):
    '''
    Test function
    '''
    print('-' * 13)
    print('    TEST')
    print('-' * 13)

    output         = ''
    checkpoint     = torch.load('../models/{}-{}/model.pt'.format(language, model_id))
    config         = checkpoint['config']

    if not torch.cuda.is_available():
        device     = torch.device('cpu')
    else:
        device     = torch.device(config['device'])

    test_loader, d = initialize_dataloader(run_type='test', language=config['language'],
                                            batch_size=1, shuffle=False)
    idx_2_char     = d.idx_2_char

    model          = initialize_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    for i_batch, sample_batched in enumerate(test_loader):

        with torch.no_grad():
            x_s = sample_batched['source_form'].to(device)
            x_t = sample_batched['target_form'].to(device)
            y_t = sample_batched['msd'].to(device)

            x_s = torch.transpose(x_s, 0, 1)
            x_t = torch.transpose(x_t, 0, 1)
            y_t = torch.transpose(y_t, 0, 1)

            x_t_p, _, _ = model(x_s, x_t, y_t)
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

    with open('../results/{}-{}-guesses'.format(config['language'], model_id), 'w+', encoding="utf-8") as f:
        f.write(output)


def train(config):
    '''
    Train function
    '''
    # Get train_dataloader
    train_loader, morph_dat = initialize_dataloader(run_type='train', language=config['language'],
                                                     batch_size=config['batch_size'], shuffle=True)

    config['vocab_size']    = morph_dat.get_vocab_size()
    config['padding_idx']   = morph_dat.padding_idx
    config['max_seq_len']   = morph_dat.max_seq_len
    config['label_len']     = morph_dat.label_len

    if not torch.cuda.is_available():
        device    = torch.device('cpu')
    else:
        device    = torch.device(config['device'])

    # Model declaration
    model         = initialize_model(config)
    optimizer     = optim.SGD(model.parameters(), lr=config['lr'])
    loss_function = nn.CrossEntropyLoss()

    kl_weight     = config['kl_start']
    anneal_rate   = (1.0 - config['kl_start']) / (config['epochs'] * len(train_loader))

    config['anneal_rate'] = anneal_rate

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

    try:
        os.mkdir('../models/{}-{}'.format(config['language'], config['model_id']))
    except OSError:
        print("Directory already exists")
        return

    epoch_details = 'epoch, bce_loss, kl_div, kl_weight, loss\n'

    model.train()
    for epoch in range(config['epochs']):

        start      = timer()
        epoch_loss = 0

        for i_batch, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            x_s = sample_batched['source_form'].to(device)
            y_t = sample_batched['msd'].to(device)
            x_t = sample_batched['target_form'].to(device)

            x_s = torch.transpose(x_s, 0, 1)
            y_t = torch.transpose(y_t, 0, 1)
            x_t = torch.transpose(x_t, 0, 1)

            x_t_p, mu, logvar = model(x_s, x_t, y_t)

            x_t_p = x_t_p[1:].view(-1, x_t_p.shape[-1])
            x_t   = x_t[1:].contiguous().view(-1)

            bce_loss = loss_function(x_t_p, x_t)
            kl_term  = kl_div(mu, logvar)

            loss     = bce_loss + kl_weight * kl_term
            loss.backward()
            optimizer.step()

            epoch_loss  += loss.detach().cpu().item()
            epoch_details += '{}, {}, {}, {}, {}\n'.format(epoch,
                                                           bce_loss.detach().cpu().item(),
                                                           kl_term.detach().cpu().item(),
                                                           kl_weight,
                                                           loss.detach().cpu().item())
            kl_weight    = min(config['lambda_m'], kl_weight + anneal_rate)

        end = timer()

        print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss / (i_batch + 1)))
        print('         - Time: {}'.format(timedelta(seconds=end - start)))

        torch.save(
            {
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss'                : epoch_loss / (i_batch + 1),
                'config'              : config
            }, '../models/{}-{}/model.pt'.format(config['language'], config['model_id']))

    with open('../models/{}-{}/epoch_details.csv'.format(config['language'], config['model_id']), 'w+') as f:
        f.write(epoch_details)


if __name__ == "__main__":

    parser                  = argparse.ArgumentParser()
    parser.add_argument('--train',       action='store_true')
    parser.add_argument('--test',        action='store_true')
    parser.add_argument('-model_id',     action="store", type=int)
    parser.add_argument('-language',     action="store", type=str)
    parser.add_argument('-device',       action="store", type=str,   default='cuda')
    parser.add_argument('-epochs',       action="store", type=int,   default=50)
    parser.add_argument('-enc_h_dim',    action="store", type=int,   default=256)
    parser.add_argument('-dec_h_dim',    action="store", type=int,   default=512)
    parser.add_argument('-char_emb_dim', action="store", type=int,   default=300)
    parser.add_argument('-tag_emb_dim',  action="store", type=int,   default=200)
    parser.add_argument('-enc_dropout',  action="store", type=float, default=0.4)
    parser.add_argument('-dec_dropout',  action="store", type=float, default=0.4)
    parser.add_argument('-z_dim',        action="store", type=int,   default=150)
    parser.add_argument('-batch_size',   action="store", type=int,   default=64)
    parser.add_argument('-kl_start',     action="store", type=float, default=0.0)
    parser.add_argument('-lambda_m',     action="store", type=float, default=0.2)
    parser.add_argument('-lr',           action="store", type=float, default=0.1)

    args                    = parser.parse_args()
    run_train               = args.train
    run_test                = args.test

    config                  = {}
    config['model_id']      = args.model_id
    config['language']      = args.language
    config['epochs']        = args.epochs
    config['kl_start']      = args.kl_start
    config['enc_h_dim']     = args.enc_h_dim
    config['dec_h_dim']     = args.dec_h_dim
    config['char_emb_dim']  = args.char_emb_dim
    config['tag_emb_dim']   = args.tag_emb_dim
    config['enc_dropout']   = args.enc_dropout
    config['dec_dropout']   = args.dec_dropout
    config['z_dim']         = args.z_dim
    config['lambda_m']      = args.lambda_m
    config['batch_size']    = args.batch_size
    config['device']        = args.device
    config['lr']            = args.lr

    # TRAIN
    if run_train:
        train(config)

    # TEST
    if run_test:
        test(config['language'], config['model_id'])
