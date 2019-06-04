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
# from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from datetime import timedelta

from models import WordEncoder, Attention, TagEmbedding, WordDecoder, MSVED, KumaMSD
from dataset import MorphologyDatasetTask3, Vocabulary

from kumaraswamy import Kumaraswamy
from hard_kumaraswamy import StretchedAndRectifiedDistribution as HardKumaraswamy


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

    encoder_x_t   = WordEncoder(input_dim   =config['vocab_size'],
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
                                device      =device,
                                no_attn     =config['no_attn'])

    model         = MSVED(encoder           =encoder,
                          tag_embedding     =tag_embedding,
                          decoder           =decoder,
                          max_len           =config['max_seq_len'],
                          vocab_size        =config['vocab_size'],
                          device            =device).to(device)

    kumaMSD       = KumaMSD(input_dim       =config['enc_h_dim'] * 2,
                            h_dim           =config['msd_h_dim'],
                            num_tags        =config['label_len'],
                            encoder         =encoder_x_t).to(device)

    return model, kumaMSD


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
    morph_data = MorphologyDatasetTask3(test=False, language=language, vocab=vocab, tasks=tasks)

    return morph_data.max_seq_len


def kl_div_sup(mu, logvar):
    '''
    Compute KL divergence between N(mu, logvar) and N(0, 1)
    '''
    return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def test(language, model_id, dont_save):
    '''
    Test function
    '''
    print('-' * 13)
    print('    TEST')
    print('-' * 13)

    output         = ''
    checkpoint     = torch.load('../models/{}-{}/model.pt'.format(language, model_id))
    config         = checkpoint['config']
    vocab          = checkpoint['vocab']

    if not torch.cuda.is_available():
        device     = torch.device('cpu')
    else:
        device     = torch.device(config['device'])

    test_loader, d = initialize_dataloader(run_type='test', language=config['language'], task='sup',
                                           vocab=vocab, batch_size=1, shuffle=False, num_workers=config['num_workers'])
    idx_2_char     = d.idx_2_char

    model, kumaMSD = initialize_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    kumaMSD.load_state_dict(checkpoint['kumaMSD_state_dict'])

    model.eval()
    kumaMSD.eval()
    for i_batch, sample_batched in enumerate(test_loader):

        with torch.no_grad():
            x_s = sample_batched['source_form'].to(device)
            x_t = sample_batched['target_form'].to(device)
            y_t = sample_batched['msd'].to(device)

            x_s = torch.transpose(x_s, 0, 1)
            x_t = torch.transpose(x_t, 0, 1)
            y_t = torch.transpose(y_t, 0, 1)

            x_t_p, _, _ = model(x_s, x_t, y_t)
            sample, _   = kumaMSD(x_t)

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

    with open('../results/{}-{}-guesses'.format(config['language'], model_id), 'w+', encoding="utf-8") as f:
        f.write(output)


def process_unsup_msd(batch_y_t, device):
    y_t_len = batch_y_t.size(0)
    output  = []

    for i in range(y_t_len):
        one_hot = [0.] * y_t_len
        one_hot[i] = batch_y_t[i]
        output.append(one_hot)

    return torch.tensor(output).to(device)


def train(config, vocab, dont_save):
    '''
    Train function
    '''
    # Writer will output to ./runs/ directory by default
    # writer = SummaryWriter(log_dir='../runs/')

    # Get train_dataloader
    train_loader_sup, morph_dat = initialize_dataloader(run_type='train', language=config['language'], task='sup',
                                                        vocab=vocab, batch_size=config['batch_size'], shuffle=True)

    if not config['only_sup']:
        train_loader_unsup, _   = initialize_dataloader(run_type='train', language=config['language'], task='unsup',
                                                        vocab=vocab, batch_size=config['batch_size'], shuffle=True, max_unsup=config['max_unsup'])

    config['vocab_size']    = morph_dat.get_vocab_size()
    config['padding_idx']   = morph_dat.padding_idx
    config['max_seq_len']   = morph_dat.max_seq_len
    config['label_len']     = len(morph_dat.desc_2_idx)

    if not torch.cuda.is_available():
        device      = torch.device('cpu')
    else:
        device      = torch.device(config['device'])

    # Model declaration
    model, kumaMSD  = initialize_model(config)

    if config['only_sup']:
        params      = model.parameters()
    else:
        params      = list(model.parameters()) + list(kumaMSD.parameters())
    optimizer       = optim.Adadelta(params,   lr=config['lr'], rho=config['rho'])
    ce_loss_func    = nn.CrossEntropyLoss()  #ignore_index=config['padding_idx'])
    loss_func_sup   = nn.BCELoss(reduction='mean')

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
    print(kumaMSD)
    print('-' * 13)
    print('    TRAIN')
    print('-' * 13)

    if not dont_save:
        try:
            os.mkdir('../models/{}-{}'.format(config['language'], config['model_id']))
        except OSError:
            print("Directory/Model already exists")
            return

    epoch_details  = 'epoch, ce_loss_sup, kl_sup, clamp_kl_sup, kl_kuma_sup, yt_loss_sup, '
    epoch_details +=        'ce_loss_unsup, kl_unsup, clamp_kl_unsup, kl_kuma_unsup, kl_kuma_unsup, '
    epoch_details += 'loss_sup, loss_unsup, total_loss\n'

    # init kuma prior
    a0 = torch.tensor([[config['a0']] * config['label_len']]).to(device)
    b0 = torch.tensor([[config['b0']] * config['label_len']]).to(device)

    kuma_prior   = Kumaraswamy(a0, b0)
    h_kuma_prior = HardKumaraswamy(kuma_prior)

    model.train()
    kumaMSD.train()
    for epoch in range(config['epochs']):

        start        = timer()
        epoch_loss   = 0
        num_batches  = 0
        count        = 1 if config['only_sup'] else 0
        done_sup     = False
        done_unsup   = False

        it_sup       = iter(train_loader_sup)
        if not config['only_sup']:
            it_unsup = iter(train_loader_unsup)

        while True:
            if count == 2:
                break

            if done_sup:
                count += 1
                it_sup = iter(train_loader_sup)

            if not config['only_sup']:
                if done_unsup:
                    count += 1
                    it_unsup = iter(train_loader_unsup)

            # Sample data points
            try:
                sample_batched_sup = next(it_sup)
            except StopIteration:
                done_sup = True
                continue

            if not config['only_sup']:
                try:
                    sample_batched_unsup = next(it_unsup)
                except StopIteration:
                    done_unsup = True
                    continue

            with autograd.detect_anomaly():
                x_s_sup   = sample_batched_sup['source_form'].to(device)
                x_t_sup   = sample_batched_sup['target_form'].to(device)
                y_t_sup   = sample_batched_sup['msd'].to(device)

                x_s_sup   = torch.transpose(x_s_sup, 0, 1)
                x_t_sup   = torch.transpose(x_t_sup, 0, 1)
                y_t_sup   = torch.transpose(y_t_sup, 0, 1)

                if not config['only_sup']:
                    x_s_unsup = sample_batched_unsup['source_form'].to(device)
                    x_t_unsup = sample_batched_unsup['target_form'].to(device)

                    x_s_unsup = torch.transpose(x_s_unsup, 0, 1)
                    x_t_unsup = torch.transpose(x_t_unsup, 0, 1)

                optimizer.zero_grad()

                ############ SUPERVISED PIPIELINE ############
                y_t_p_sup, h_kuma_post_sup    = kumaMSD(x_t_sup)
                x_t_p_sup, mu_sup, logvar_sup = model(x_s_sup, x_t_sup, y_t_sup)

                x_t_p_sup = x_t_p_sup[1:].view(-1, x_t_p_sup.shape[-1])
                x_t_a_sup = x_t_sup[1:].contiguous().view(-1)

                # Compute supervised loss
                ce_loss_sup   = ce_loss_func(x_t_p_sup, x_t_a_sup)
                kl_sup        = kl_div_sup(mu_sup, logvar_sup)
                # kuma_loss_sup = torch.sum(torch.distributions.kl.kl_divergence(h_kuma_post_sup, h_kuma_prior))
                kuma_loss_sup = h_kuma_prior.log_prob(torch.sum(y_t_sup, dim=0))
                # yt_loss_sup = loss_func_sup(y_t_p_sup, torch.sum(y_t_sup, dim=0))

                # ha bits, like free bits but over whole layer
                # REFERENCE: https://github.com/kastnerkyle/pytorch-text-vae
                habits_lambda = config['lambda_m']
                clamp_kl_sup  = torch.clamp(kl_sup.mean(), min=habits_lambda).squeeze()
                loss_sup      = ce_loss_sup + kl_sup * clamp_kl_sup + kuma_loss_sup # + yt_loss_sup

                ############ UNSUPERVISED PIPIELINE ############
                loss_unsup     = torch.zeros(1).to(device)
                ce_loss_unsup  = torch.zeros(1).to(device)
                kl_unsup       = torch.zeros(1).to(device)
                clamp_kl_unsup = torch.zeros(1).to(device)
                kl_kuma_unsup  = torch.zeros(1).to(device)

                if not config['only_sup']:
                    y_t_p_unsup, h_kuma_post_unsup      = kumaMSD(x_t_unsup)
                    y_t_unsup                           = torch.stack([process_unsup_msd(batch, device) for batch in y_t_p_unsup])
                    y_t_unsup                           = y_t_unsup.permute(1, 0, 2)
                    x_t_p_unsup, mu_unsup, logvar_unsup = model(x_s_unsup, x_t_unsup, y_t_unsup)

                    x_t_p_unsup = x_t_p_unsup[1:].view(-1, x_t_p_unsup.shape[-1])
                    x_t_a_unsup = x_t_unsup[1:].contiguous().view(-1)

                    # Compute unsupervised loss
                    ce_loss_unsup  = ce_loss_func(x_t_p_unsup, x_t_a_unsup)
                    kl_unsup       = kl_div_sup(mu_unsup, logvar_unsup)
                    kl_kuma_unsup  = torch.sum(torch.distributions.kl.kl_divergence(h_kuma_post_unsup, h_kuma_prior))

                    clamp_kl_unsup = torch.clamp(kl_unsup.mean(), min=habits_lambda).squeeze()
                    loss_unsup     = ce_loss_unsup + kl_unsup * clamp_kl_unsup + kl_kuma_unsup

                total_loss     = loss_sup + config['dt_unsup'] * loss_unsup
                total_loss.backward()

            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(kumaMSD.parameters()), 10)
            optimizer.step()

            num_batches   += 1
            epoch_loss    += total_loss.detach().cpu().item()
            epoch_details += '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                                epoch,
                                ce_loss_sup.detach().cpu().item(),
                                kl_sup.detach().cpu().item(),
                                clamp_kl_sup.detach().cpu().item(),
                                kl_kuma_sup.detach().cpu().item(),
                                yt_loss_sup.detach().cpu().item(),
                                ce_loss_unsup.detach().cpu().item(),
                                kl_unsup.detach().cpu().item(),
                                clamp_kl_unsup.detach().cpu().item(),
                                kl_kuma_unsup.detach().cpu().item(),
                                loss_sup.detach().cpu().item(),
                                loss_unsup.detach().cpu().item(),
                                total_loss.detach().cpu().item())

            kl_weight     = min(1.0, kl_weight + anneal_rate)

        end = timer()

        print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss / (num_batches + 1)))
        print('         - Time: {}'.format(timedelta(seconds=end - start)))

        if dont_save:
            continue

        torch.save(
            {
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'kumaMSD_state_dict'  : kumaMSD.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss'                : epoch_loss / (num_batches + 1),
                'config'              : config,
                'vocab'               : vocab
            }, '../models/{}-{}/model.pt'.format(config['language'], config['model_id']))

    if dont_save:
        return

    with open('../models/{}-{}/epoch_details.csv'.format(config['language'], config['model_id']), 'w+') as f:
        f.write(epoch_details)


def continue_training(model, config):
    pass


if __name__ == "__main__":

    parser                  = argparse.ArgumentParser()
    parser.add_argument('--train',       action="store_true")
    parser.add_argument('--test',        action="store_true")
    parser.add_argument('--only_sup',    action="store_true")
    parser.add_argument('--no_attn',     action="store_true",        default=False)
    parser.add_argument('--dont_save',   action="store_true",        default=False)
    parser.add_argument('-model_id',     action="store", type=int)
    parser.add_argument('-language',     action="store", type=str)
    parser.add_argument('-device',       action="store", type=str,   default='cuda')
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
    parser.add_argument('-lr',           action="store", type=float, default=0.1)
    parser.add_argument('-kuma_msd',     action="store", type=int,   default=256)
    parser.add_argument('-a0',           action="store", type=float, default=0.139)
    parser.add_argument('-b0',           action="store", type=float, default=0.286)
    parser.add_argument('-rho',          action="store", type=float, default=0.95)
    parser.add_argument('-max_unsup',    action="store", type=int,   default=10000)
    parser.add_argument('-dt_unsup',     action="store", type=float, default=0.7)
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
    config['enc_dropout']   = args.enc_dropout
    config['dec_dropout']   = args.dec_dropout
    config['z_dim']         = args.z_dim
    config['lambda_m']      = args.lambda_m
    config['batch_size']    = args.batch_size
    config['device']        = args.device
    config['lr']            = args.lr
    config['msd_h_dim']     = args.kuma_msd
    config['a0']            = args.a0
    config['b0']            = args.b0
    config['rho']           = args.rho
    config['max_unsup']     = args.max_unsup
    config['only_sup']      = args.only_sup
    config['dt_unsup']      = args.dt_unsup
    config['num_workers']   = args.num_workers
    config['no_attn']       = args.no_attn

    # TRAIN
    if run_train:
        vocab               = Vocabulary(language=config['language'])
        train(config, vocab, dont_save)

    # TEST
    if run_test:
        test(config['language'], config['model_id'], dont_save)
