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

from models import KumaMSD, WordEncoder
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

    if config['model'] == 'kumamsd':
        encoder_x_t   = WordEncoder(input_dim   =config['vocab_size'],
                                    emb_dim     =config['char_emb_dim'],
                                    enc_h_dim   =config['enc_h_dim'],
                                    z_dim       =config['z_dim'],
                                    dropout     =0.,
                                    padding_idx =config['padding_idx'],
                                    device      =device)

        kumaMSD       = KumaMSD(input_dim       =config['enc_h_dim'] * 2,
                                h_dim           =config['msd_h_dim'],
                                num_tags        =config['label_len'],
                                encoder         =encoder_x_t,
                                device          =device,
                                unconstrained   =config['unconstrained'],
                                use_made        =config['use_made']).to(device)

        return kumaMSD


def initialize_dataloader(run_type, language, task, vocab, batch_size, shuffle, max_unsup=10000, num_workers=2):
    '''
    Initializes train and test dataloaders
    '''
    is_test     = (run_type == 'test')
    max_seq_len = get_max_seq_len(language, vocab)

    if task == 'kumamsd':
        tasks = ['task3p']

    morph_data = MorphologyDatasetTask3(test=is_test, language=language, vocab=vocab, tasks=tasks, get_unprocessed=is_test,
                                        max_unsup=max_unsup, max_seq_len=max_seq_len)
    dataloader = DataLoader(morph_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=(run_type == 'train'))

    return dataloader, morph_data


def get_max_seq_len(language, vocab):

    tasks = ['task1p', 'task2p', 'task3p']
    morph_data = MorphologyDatasetTask3(test=False, language=language, vocab=vocab, tasks=tasks)

    return morph_data.max_seq_len


def train(config, vocab, dont_save):
    '''
    Train function
    '''
    train_loader_sup, morph_dat = initialize_dataloader(run_type='train', language=config['language'], task=config['model'],
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
    ce_loss         = nn.BCELoss()
    len_data        = len(train_loader_sup)

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

    epoch_details  = 'epoch, loss, bceloss\n'

    model.train()
    for epoch in range(config['epochs']):

        start        = timer()
        epoch_loss   = 0
        num_batches  = 0

        it_sup       = iter(train_loader_sup)

        while True:
            try:
                sample_batched_sup = next(it_sup)
            except StopIteration:
                break

            with autograd.detect_anomaly():
                x_t   = sample_batched_sup['target_form'].to(device)
                y_t   = sample_batched_sup['msd'].to(device)
                x_t   = torch.transpose(x_t, 0, 1)
                y_t   = torch.transpose(y_t, 0, 1)

                optimizer.zero_grad()

                ############ PIPIELINE ############
                y_t_p, h_kuma_post  = model(x_t)

                # Compute supervised loss
                bce_loss   = ce_loss(y_t_p, torch.sum(y_t, dim=0))
                total_loss = -1. * torch.mean(h_kuma_post.log_prob(torch.sum(y_t, dim=0)))
                total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            num_batches   += 1
            epoch_loss    += total_loss.detach().cpu().item()
            epoch_details += '{}, {}, {}\n'.format(epoch, total_loss.detach().cpu().item(),
                                                   bce_loss.detach().cpu().item())

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
    parser.add_argument('--unconstrained', action="store_true",      default=False)
    parser.add_argument('--use_made',      action="store_true",      default=False)

    parser.add_argument('-model',        action="store", type=str,   default='kumamsd')
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
    parser.add_argument('-msd_h_dim',    action="store", type=int,   default=256)
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
    config['msd_h_dim']     = args.msd_h_dim
    config['unconstrained'] = args.unconstrained
    config['use_made']      = args.use_made

    # TRAIN
    if run_train:
        vocab               = Vocabulary(language=config['language'])
        train(config, vocab, dont_save)
