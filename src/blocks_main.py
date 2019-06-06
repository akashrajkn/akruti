import os
import pickle
import random
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch import autograd
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from datetime import timedelta

from blocks import VAE, L_MSVAE
from models import TagEmbedding
from dataset import MorphologyDatasetTask3, Vocabulary


np.random.seed(0)
torch.manual_seed(0)


def initialize_model(config):
    '''
    Initialize and return the models
    Args:
        config -- config dict
    Return:
        model
    '''
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(config['device'])

    if config['model'] == 'vae':
        vae        = VAE(vocab_size  =config['vocab_size'],
                         emb_dim     =config['char_emb_dim'],
                         z_dim       =config['z_dim'],
                         h_dim       =config['enc_h_dim'],
                         device      =device,
                         max_len     =config['max_seq_len'],
                         padding_idx =config['padding_idx']).to(device)
        return vae

    if config['model'] == 'msvae':
        tag_embedding = TagEmbedding(input_dim  =config['label_len'],
                                     emb_dim    =config['tag_emb_dim'],
                                     device     =device)

        msvae         = L_MSVAE(vocab_size   =config['vocab_size'],
                                emb_dim      =config['char_emb_dim'],
                                tag_embedding=tag_embedding,
                                z_dim        =config['z_dim'],
                                h_dim        =config['enc_h_dim'],
                                device       =device,
                                max_len      =config['max_seq_len'],
                                padding_idx  =config['padding_idx']).to(device)

        return msvae


def initialize_dataloader(run_type, language, task, vocab, batch_size, shuffle, max_unsup=10000, num_workers=2):
    '''
    Initializes train and test dataloaders
    '''
    is_test    = (run_type == 'test')

    max_seq_len = get_max_seq_len(language, vocab)

    if task == 'sup':
        tasks = ['task3p']
    elif task == 'MVSAE':
        tasks = ['task1', 'task3p']
    else:
        tasks = ['task1p', 'task2p']

    morph_data = MorphologyDatasetTask3(test=is_test, language=language, vocab=vocab, tasks=tasks, get_unprocessed=is_test,
                                        max_unsup=max_unsup, max_seq_len=max_seq_len)
    dataloader = DataLoader(morph_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=(run_type == 'train'))

    return dataloader, morph_data


def get_max_seq_len(language, vocab):

    tasks = ['task1p', 'task2p', 'task3p']
    morph_data = MorphologyDatasetTask3(test=False, language=language, vocab=vocab, tasks=tasks)

    return morph_data.max_seq_len


def kl_div_sup(mu, logvar):
    '''
    Compute KL divergence between N(mu, logvar) and N(0, 1)
    '''
    return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def train(config, vocab, dont_save):
    '''
    Train function
    '''
    train_loader_sup, morph_dat = initialize_dataloader(run_type='train', language=config['language'], task='MVSAE',
                                                        vocab=vocab, batch_size=config['batch_size'], shuffle=True)

    config['vocab_size']    = morph_dat.get_vocab_size()
    config['padding_idx']   = morph_dat.padding_idx
    config['max_seq_len']   = morph_dat.max_seq_len
    config['label_len']     = len(morph_dat.desc_2_idx)

    if not torch.cuda.is_available():
        device      = torch.device('cpu')
    else:
        device      = torch.device(config['device'])

    # Model declaration
    model           = initialize_model(config)
    optimizer       = optim.Adadelta(model.parameters(), lr=config['lr'], rho=config['rho'])
    ce_loss_func    = nn.CrossEntropyLoss()  #ignore_index=config['padding_idx'])

    kl_weight       = config['kl_start']
    len_data        = len(train_loader_sup)
    anneal_rate     = (1.0 - config['kl_start']) / (config['epochs'] * (len_data))

    config['anneal_rate'] = anneal_rate

    print('-' * 13)
    print('  DEVICE:   {}'.format(device))
    print('  LANGUAGE: {}'.format(config['language']))
    print('-' * 13)
    print('    MODEL')
    print('-' * 13)
    print(model)
    print('-' * 13)
    print('    TRAIN')
    print('-' * 13)

    if not dont_save:
        try:
            os.mkdir('../models/{}_{}-{}'.format(config['model'], config['language'], config['model_id']))
        except OSError:
            print("Directory/Model already exists")
            return

    epoch_details  = 'epoch, ce_loss_sup, kl_sup, clamp_kl_sup, loss_sup\n'

    model.train()
    for epoch in range(config['epochs']):

        start        = timer()
        epoch_loss   = 0
        num_batches  = 0

        it_sup       = iter(train_loader_sup)

        while True:
            # Sample data points
            try:
                sample_batched_sup = next(it_sup)
            except StopIteration:
                break

            with autograd.detect_anomaly():
                x_s_sup   = sample_batched_sup['source_form'].to(device)
                x_t_sup   = sample_batched_sup['target_form'].to(device)
                y_t_sup   = sample_batched_sup['msd'].to(device)

                x_s_sup   = torch.transpose(x_s_sup, 0, 1)
                x_t_sup   = torch.transpose(x_t_sup, 0, 1)
                y_t_sup   = torch.transpose(y_t_sup, 0, 1)

                optimizer.zero_grad()

                ############ PIPIELINE ############
                x_t_p_sup, mu_sup, logvar_sup = model(x_t_sup, y_t_sup)

                x_t_p_sup = x_t_p_sup[1:].view(-1, x_t_p_sup.shape[-1])
                x_t_a_sup = x_t_sup[1:].contiguous().view(-1)

                # Compute supervised loss
                ce_loss_sup   = ce_loss_func(x_t_p_sup, x_t_a_sup)
                kl_sup        = kl_div_sup(mu_sup, logvar_sup)

                # ha bits, like free bits but over whole layer
                # REFERENCE: https://github.com/kastnerkyle/pytorch-text-vae
                habits_lambda = config['lambda_m']
                clamp_kl_sup  = torch.clamp(kl_sup.mean(), min=habits_lambda).squeeze()
                loss_sup      = ce_loss_sup + kl_sup * clamp_kl_sup

                total_loss    = loss_sup
                total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            num_batches   += 1
            epoch_loss    += total_loss.detach().cpu().item()
            epoch_details += '{}, {}, {}, {}, {}\n'.format(
                                epoch,
                                ce_loss_sup.detach().cpu().item(),
                                kl_sup.detach().cpu().item(),
                                clamp_kl_sup.detach().cpu().item(),
                                loss_sup.detach().cpu().item())

            kl_weight      = min(1.0, kl_weight + anneal_rate)

        end = timer()

        print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss / (num_batches + 1)))
        print('         - Time: {}'.format(timedelta(seconds=end - start)))

        if dont_save:
            continue

        torch.save(
            {
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss'                : epoch_loss / (num_batches + 1),
                'config'              : config,
                'vocab'               : vocab
            }, '../models/{}_{}-{}/model.pt'.format(config['model'], config['language'], config['model_id']))

    if dont_save:
        return

    with open('../models/{}_{}-{}/epoch_details.csv'.format(config['model'], config['language'], config['model_id']), 'w+') as f:
        f.write(epoch_details)


if __name__ == "__main__":

    parser                  = argparse.ArgumentParser()
    parser.add_argument('--train',       action="store_true")
    parser.add_argument('--test',        action="store_true")
    parser.add_argument('--dont_save',   action="store_true",        default=False)
    parser.add_argument('-model',        action="store", type=str,   default='vae')
    parser.add_argument('-model_id',     action="store", type=int)
    parser.add_argument('-language',     action="store", type=str)
    parser.add_argument('-device',       action="store", type=str,   default='cuda')
    parser.add_argument('-epochs',       action="store", type=int,   default=50)
    parser.add_argument('-enc_h_dim',    action="store", type=int,   default=256)
    parser.add_argument('-dec_h_dim',    action="store", type=int,   default=512)
    parser.add_argument('-char_emb_dim', action="store", type=int,   default=300)
    parser.add_argument('-tag_emb_dim',  action="store", type=int,   default=200)
    parser.add_argument('-z_dim',        action="store", type=int,   default=150)
    parser.add_argument('-batch_size',   action="store", type=int,   default=32)
    parser.add_argument('-kl_start',     action="store", type=float, default=0.0)
    parser.add_argument('-lambda_m',     action="store", type=float, default=0.2)
    parser.add_argument('-lr',           action="store", type=float, default=0.1)
    parser.add_argument('-rho',          action="store", type=float, default=0.95)
    parser.add_argument('-num_workers',  action="store", type=int,   default=2)

    args                    = parser.parse_args()
    run_train               = args.train
    run_test                = args.test
    dont_save               = args.dont_save

    config                  = {}
    config['model_id']      = args.model_id
    config['language']      = args.language
    config['epochs']        = args.epochs
    config['kl_start']      = args.kl_start
    config['enc_h_dim']     = args.enc_h_dim
    config['dec_h_dim']     = args.dec_h_dim
    config['char_emb_dim']  = args.char_emb_dim
    config['tag_emb_dim']   = args.tag_emb_dim
    config['z_dim']         = args.z_dim
    config['lambda_m']      = args.lambda_m
    config['batch_size']    = args.batch_size
    config['device']        = args.device
    config['lr']            = args.lr
    config['rho']           = args.rho
    config['num_workers']   = args.num_workers
    config['model']         = args.model

    # TRAIN
    if run_train:
        vocab               = Vocabulary(language=config['language'])
        train(config, vocab, dont_save)
