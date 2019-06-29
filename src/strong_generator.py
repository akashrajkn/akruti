import random
import math
import time
import os
import pickle
import logging
import argparse

import numpy as np

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from datetime import timedelta

from dataset import MorphologyDatasetTask3, Vocabulary

np.random.seed(0)
torch.manual_seed(0)


class WordEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim=300, enc_h_dim=256, z_dim=150,
                 dropout=0.0, padding_idx=None, device=None):
        '''
        input_dim   -- vocabulary(characters) size
        emb_dim     -- character embedding dimension
        enc_h_dim -- RNN hidden state dimenion
        z_dim       -- latent variable z dimension
        '''
        super(WordEncoder, self).__init__()

        self.input_dim   = input_dim
        self.emb_dim     = emb_dim
        self.enc_h_dim   = enc_h_dim
        self.z_dim       = z_dim
        self.device      = device
        self.padding_idx = padding_idx

        self.embedding   = nn.Embedding(input_dim, emb_dim, padding_idx=padding_idx)
        self.rnn         = nn.GRU(emb_dim, enc_h_dim, bidirectional=True)
        self.fc_mu       = nn.Linear(enc_h_dim * 2, z_dim)
        self.fc_sigma    = nn.Linear(enc_h_dim * 2, z_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, src):
        embedded         = self.dropout(self.embedding(src))
        _, hidden        = self.rnn(embedded)
        hidden           = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        z_mu             = F.relu(self.fc_mu(hidden))
        z_logvar         = F.relu(self.fc_sigma(hidden))

        return hidden, z_mu, z_logvar


class WordDecoder(nn.Module):
    def __init__(self, attention, output_dim, char_emb_dim=300, z_dim=150, tag_emb_dim=200,
                 dec_h_dim=512, dropout=0.4, padding_idx=None, device=None, no_attn=False):
        '''
        output_dim -- vocabulary size
        '''
        super(WordDecoder, self).__init__()

        self.z_dim        = z_dim
        self.char_emb_dim = char_emb_dim
        self.tag_emb_dim  = tag_emb_dim
        self.dec_h_dim    = dec_h_dim
        self.output_dim   = output_dim
        self.attention    = attention
        self.device       = device
        self.no_attn      = no_attn

        self.embedding    = nn.Embedding(output_dim, char_emb_dim)
        self.rnn          = nn.GRU(char_emb_dim + z_dim, dec_h_dim)
        self.out          = nn.Linear(dec_h_dim, output_dim)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, input, hidden, z, drop):
        '''
        input      -- previous input
        hidden     -- hidden state of the decoder
        z          -- lemma represented by the latent variable
        drop       -- drop character
        '''
        input    = input.type(torch.LongTensor).to(self.device)
        input    = input.unsqueeze(0)
        embedded = self.embedding(input)
        z        = z.unsqueeze(0)  # For tensor dimension compatibility - see rnn_input

        if drop:
            embedded = embedded * 0.

        rnn_input      = torch.cat((embedded, z), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        output         = self.out(output.squeeze(0))

        return output, hidden.squeeze(0)


class MSVED(nn.Module):
    '''
    Multi-space Variational Encoder-Decoder
    '''
    def __init__(self, encoder, tag_embedding, decoder, max_len, vocab_size, device, dropout_type='random_chars'):
        super(MSVED, self).__init__()

        self.encoder       = encoder
        self.tag_embedding = tag_embedding
        self.decoder       = decoder

        self.max_len       = max_len
        self.device        = device
        self.vocab_size    = vocab_size
        self.dropout_type  = dropout_type

        # dropout: 40%
        self.dropout_dist  = dist.bernoulli.Bernoulli(torch.tensor([0.4]))
        self.poisson_dist  = dist.poisson.Poisson(torch.tensor([3.5]))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x_s, x_t):

        batch_size     = x_s.shape[1]
        outputs        = torch.zeros(self.max_len, batch_size, self.vocab_size).to(self.device)

        h, mu_u, var_u = self.encoder(x_s)
        # z              = self.reparameterize(mu_u, var_u)
        o              = x_t[0, :]  # Start tokens

        dropout_idx = []

        if self.dropout_type == 'contiguous':
            mi = []
            while True:
                if len(mi) == 2:  # FIXME: make this a hyper parameter
                    break

                idx = np.random.choice(range(self.max_len))
                if idx == 0:
                    continue
                mi.append(idx)

            for i in mi:
                length = int(self.poisson_dist.sample().item())
                for j in range(i, i + length):
                    dropout_idx.append(j)

        for t in range(1, self.max_len):

            drop = False

            if self.dropout_type   == 'random_chars':
                drop     = (self.dropout_dist.sample() == torch.tensor([1.]))
            elif self.dropout_type == 'contiguous':
                if t in dropout_idx:
                    drop = True

            o, h       = self.decoder(o, h, mu_u, drop)
            outputs[t] = o
            o          = o.max(1)[1]

        return outputs, mu_u, var_u


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

    decoder       = WordDecoder(attention   =None,
                                output_dim  =config['vocab_size'],
                                char_emb_dim=config['char_emb_dim'],
                                z_dim       =config['z_dim'],
                                tag_emb_dim =config['tag_emb_dim'],
                                dec_h_dim   =config['dec_h_dim'],
                                dropout     =config['dec_dropout'],
                                padding_idx =config['padding_idx'],
                                device      =device,
                                no_attn     =config['no_attn'])

    model         = MSVED(encoder           =encoder,
                          tag_embedding     =None,
                          decoder           =decoder,
                          max_len           =config['max_seq_len'],
                          vocab_size        =config['vocab_size'],
                          device            =device,
                          dropout_type      =config['dropout_type']).to(device)

    return model


def initialize_dataloader(run_type, language, task, vocab, batch_size, shuffle, max_unsup=10000, num_workers=2):
    '''
    Initializes train and test dataloaders
    '''
    is_test    = (run_type == 'test')

    max_seq_len = get_max_seq_len(language, vocab)

    if task == 'sup':
        tasks = ['task3p']
    else:
        tasks = ['task1p', 'task2p']

    morph_data = MorphologyDatasetTask3(test=is_test, language=language, vocab=vocab, tasks=tasks, get_unprocessed=is_test,
                                        max_unsup=max_unsup, max_seq_len=max_seq_len)
    dataloader = DataLoader(morph_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=(run_type == 'train'))

    return dataloader, morph_data


def get_max_seq_len(language, vocab):

    tasks = ['task1p', 'task2p', 'task3p']
    morph_data = MorphologyDatasetTask3(test=False, language=language, vocab=vocab, tasks=tasks, for_max_len=True)

    return morph_data.max_seq_len


def kl_div_sup(mu, logvar):
    '''
    Compute KL divergence between N(mu, logvar) and N(0, 1)
    '''
    # return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return - 0.5 * (2 * logvar - torch.pow(mu, 2) - torch.pow(torch.exp(logvar), 2) + 1)


def test(language, model_id, dont_save):
    '''
    Test function
    '''
    print('-' * 13)
    print('    TEST')
    print('-' * 13)

    output         = ''
    checkpoint     = torch.load('../models/gen_{}-{}/model.pt'.format(language, model_id))
    config         = checkpoint['config']
    vocab          = checkpoint['vocab']

    if not torch.cuda.is_available():
        device     = torch.device('cpu')
    else:
        device     = torch.device(config['device'])

    test_loader, d = initialize_dataloader(run_type='test', language=config['language'], task='sup',
                                           vocab=vocab, batch_size=1, shuffle=False, num_workers=config['num_workers'])
    idx_2_char     = d.idx_2_char

    model          = initialize_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    for i_batch, sample_batched in enumerate(test_loader):

        with torch.no_grad():
            x_s = sample_batched['source_form'].to(device)
            x_t = sample_batched['target_form'].to(device)

            x_s = torch.transpose(x_s, 0, 1)
            x_t = torch.transpose(x_t, 0, 1)

            x_t_p, _, _ = model(x_s, x_t)
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

    if dont_save:
        return

    with open('../results/gen_{}-{}-guesses'.format(config['language'], model_id), 'w+', encoding="utf-8") as f:
        f.write(output)


def process_unsup_msd(batch_y_t, device):
    y_t_len = batch_y_t.size(0)
    output  = []

    for i in range(y_t_len):
        one_hot = [0.] * y_t_len
        one_hot[i] = batch_y_t[i]
        output.append(one_hot)

    return torch.tensor(output).to(device)


def compute_supervised_loss(ce_loss_func, x_t_p_sup, x_t_a_sup, mu_sup, logvar_sup,
                            kl_weight, lag_weight, config):
    '''Compute supervised loss'''

    ce_loss_sup      = ce_loss_func(x_t_p_sup, x_t_a_sup)
    kl_sup           = kl_div_sup(mu_sup, logvar_sup)
    kuma_loss_sup    = torch.zeros(1)
    yt_loss_sup      = torch.zeros(1)

    if   config['elbo_fix'] == 'kl_anneal':
        clamp_kl_sup         = kl_weight * kl_sup
    elif config['elbo_fix'] == 'habits':
        # ha bits, like free bits but over whole layer
        # REFERENCE: https://github.com/kastnerkyle/pytorch-text-vae
        habits_lambda        = config['lambda_m']
        clamp_kl_sup         = torch.clamp(kl_sup.mean(), min=habits_lambda).squeeze()
    elif config['elbo_fix'] == 'mdr':
        rate                 = config['lambda_m']
        clamp_kl_sup         = kl_sup.mean() + lag_weight.abs() * (rate - kl_sup.mean())

    loss_sup = ce_loss_sup + clamp_kl_sup

    return ce_loss_sup, kl_sup, kuma_loss_sup, yt_loss_sup, clamp_kl_sup, loss_sup


def train(config, vocab, dont_save):
    '''
    Train function
    '''
    # Writer will output to ./runs/ directory by default
    # writer = SummaryWriter(log_dir='../runs/')

    # Get train_dataloader
    train_loader_sup, morph_dat = initialize_dataloader(run_type='train', language=config['language'], task='sup',
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

    if config['only_sup']:
        params      = model.parameters()
    else:
        params      = list(model.parameters()) + list(kumaMSD.parameters())
    optimizer       = optim.Adadelta(params,   lr=config['lr'], rho=config['rho'])
    lag_weight      = torch.rand(1, requires_grad=True, device=device)
    optimizer_mdr   = optim.RMSprop([lag_weight])
    ce_loss_func    = nn.CrossEntropyLoss(ignore_index=config['padding_idx'])

    kl_weight       = config['kl_start']

    if config['only_sup']:
        len_data    = len(train_loader_sup)
    else:
        len_data    = len(train_loader_sup) + len(train_loader_unsup)
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
            os.mkdir('../models/gen_{}-{}'.format(config['language'], config['model_id']))
        except OSError:
            print("Directory/Model already exists")
            return

    epoch_details  = 'epoch, ce_loss_sup, kl_sup, clamp_kl_sup, kuma_loss_sup, yt_loss_sup, '
    epoch_details += 'loss_sup, total_loss\n'

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

            if not config['only_sup']:
                try:
                    sample_batched_unsup = next(it_unsup)
                except StopIteration:
                    done_unsup = True
                    continue

            with autograd.detect_anomaly():
                x_s_sup   = sample_batched_sup['source_form'].to(device)
                x_t_sup   = sample_batched_sup['target_form'].to(device)

                x_s_sup   = torch.transpose(x_s_sup, 0, 1)
                x_t_sup   = torch.transpose(x_t_sup, 0, 1)

                optimizer.zero_grad()

                if config['elbo_fix'] == 'mdr':
                    optimizer_mdr.zero_grad()

                ############ SUPERVISED PIPIELINE ############
                x_t_p_sup, mu_sup, logvar_sup = model(x_s_sup, x_t_sup)

                x_t_p_sup = x_t_p_sup[1:].view(-1, x_t_p_sup.shape[-1])
                x_t_a_sup = x_t_sup[1:].contiguous().view(-1)

                ce_loss_sup, kl_sup, kuma_loss_sup, yt_loss_sup, clamp_kl_sup, loss_sup = compute_supervised_loss(ce_loss_func, x_t_p_sup, x_t_a_sup, mu_sup, logvar_sup,
                                                                                                                  kl_weight, lag_weight, config)

                total_loss = loss_sup
                total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            if config['elbo_fix'] == 'mdr':
                for group in optimizer_mdr.param_groups:
                    for p in group['params']:
                        p.grad = -1 * p.grad
                optimizer_mdr.step()

            optimizer.step()

            num_batches   += 1
            epoch_loss    += total_loss.detach().cpu().item()

            epoch_details += '{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                                epoch,
                                ce_loss_sup.detach().cpu().item(),
                                kl_sup.mean().detach().cpu().item(),
                                clamp_kl_sup.detach().cpu().item(),
                                kuma_loss_sup.detach().cpu().item(),
                                yt_loss_sup.detach().cpu().item(),
                                loss_sup.detach().cpu().item(),
                                total_loss.detach().cpu().item())

            kl_weight     = min(1.0, kl_weight + anneal_rate)
            break

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
            }, '../models/gen_{}-{}/model.pt'.format(config['language'], config['model_id']))

    if dont_save:
        return

    with open('../models/gen_{}-{}/epoch_details.csv'.format(config['language'], config['model_id']), 'w+') as f:
        f.write(epoch_details)


def continue_training(model, config):
    pass


if __name__ == "__main__":

    parser                  = argparse.ArgumentParser()
    parser.add_argument('--train',         action="store_true")
    parser.add_argument('--test',          action="store_true")
    parser.add_argument('--only_sup',      action="store_true")
    parser.add_argument('--no_attn',       action="store_true",      default=False)
    parser.add_argument('--dont_save',     action="store_true",      default=False)
    parser.add_argument('--unconstrained', action="store_true",      default=False)
    parser.add_argument('--use_made',      action="store_true",      default=False)

    parser.add_argument('-model_id',     action="store", type=int)
    parser.add_argument('-language',     action="store", type=str)
    parser.add_argument('-device',       action="store", type=str,   default='cuda')
    parser.add_argument('-drop_type',    action="store", type=str,   default='random_chars')
    parser.add_argument('-epochs',       action="store", type=int,   default=50)
    parser.add_argument('-enc_h_dim',    action="store", type=int,   default=256)
    parser.add_argument('-dec_h_dim',    action="store", type=int,   default=512)
    parser.add_argument('-char_emb_dim', action="store", type=int,   default=300)
    parser.add_argument('-tag_emb_dim',  action="store", type=int,   default=200)
    parser.add_argument('-enc_dropout',  action="store", type=float, default=0.0)
    parser.add_argument('-dec_dropout',  action="store", type=float, default=0.4)
    parser.add_argument('-z_dim',        action="store", type=int,   default=150)
    parser.add_argument('-batch_size',   action="store", type=int,   default=64)
    parser.add_argument('-kl_start',     action="store", type=float, default=0.0)
    parser.add_argument('-lambda_m',     action="store", type=float, default=0.2)
    parser.add_argument('-lambda_kuma',  action="store", type=float, default=0.2)
    parser.add_argument('-lr',           action="store", type=float, default=0.1)
    parser.add_argument('-kuma_msd',     action="store", type=int,   default=256)
    parser.add_argument('-rho',          action="store", type=float, default=0.95)
    parser.add_argument('-max_unsup',    action="store", type=int,   default=10000)
    parser.add_argument('-dt_unsup',     action="store", type=float, default=0.7)
    parser.add_argument('-num_workers',  action="store", type=int,   default=2)
    parser.add_argument('-elbo_fix',     action="store", type=str,   default='habits')

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
    config['enc_dropout']   = args.enc_dropout
    config['dec_dropout']   = args.dec_dropout
    config['z_dim']         = args.z_dim
    config['lambda_m']      = args.lambda_m
    config['lambda_kuma']   = args.lambda_kuma
    config['batch_size']    = args.batch_size
    config['device']        = args.device
    config['lr']            = args.lr
    config['msd_h_dim']     = args.kuma_msd
    config['rho']           = args.rho
    config['max_unsup']     = args.max_unsup
    config['only_sup']      = args.only_sup
    config['dt_unsup']      = args.dt_unsup
    config['num_workers']   = args.num_workers
    config['no_attn']       = args.no_attn
    config['unconstrained'] = args.unconstrained
    config['use_made']      = args.use_made
    config['dropout_type']  = args.drop_type
    config['elbo_fix']      = args.elbo_fix

    # TRAIN
    if run_train:
        vocab               = Vocabulary(language=config['language'])
        train(config, vocab, dont_save)

    # TEST
    if run_test:
        test(config['language'], config['model_id'], dont_save)
