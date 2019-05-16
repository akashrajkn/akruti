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

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from datetime import timedelta

from models import WordEncoder, Attention, TagEmbedding, WordDecoder, MSVED, KumaMSD
from dataset import MorphologyDatasetTask3, Vocabulary

from kumaraswamy import Kumaraswamy


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
                                device      =device)

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


def initialize_dataloader(run_type, language, task, vocab, batch_size, shuffle):
    '''
    Initializes train and test dataloaders
    '''
    is_test    = (run_type == 'test')

    if task == 'sup':
        tasks = ['task3']
    else:
        tasks = ['task1p', 'task2p']

    morph_data = MorphologyDatasetTask3(test=is_test, language=language, vocab=vocab, tasks=tasks, get_unprocessed=is_test)
    dataloader = DataLoader(morph_data, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return dataloader, morph_data


def kl_div(mu, logvar):
    '''
    Compute KL divergence between N(mu, logvar) and N(0, 1)
    '''
    return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def loss_kuma(kuma_prior, kuma_post, supervised=False):
    '''
    If supervised, computes the log probability.
    In the unsupervised case, computes the KL between the two Kuma distributions
    '''

    if supervised:
        return kuma_prior.log_prob(kuma_prior.sample())

    kl_div = -torch.distributions.kl.kl_divergence(kuma_post, kuma_prior)

    return torch.sum(kl_div)


def test(language, model_id, vocab, dont_save):
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

    test_loader, d = initialize_dataloader(run_type='test', language=config['language'], task='sup',
                                           vocab=vocab, batch_size=1, shuffle=False)
    idx_2_char     = d.idx_2_char

    model, _       = initialize_model(config)
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

    if dont_save:
        return

    with open('../results/{}-{}-guesses'.format(config['language'], model_id), 'w+', encoding="utf-8") as f:
        f.write(output)


def train(config, vocab, dont_save):
    '''
    Train function
    '''
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir='../runs/')

    # Get train_dataloader
    train_loader_sup, morph_dat = initialize_dataloader(run_type='train', language=config['language'], task='sup',
                                                        vocab=vocab, batch_size=config['batch_size'], shuffle=True)
    train_loader_unsup, _       = initialize_dataloader(run_type='train', language=config['language'], task='unsup',
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
    model, kumaMSD  = initialize_model(config)
    params          = list(model.parameters()) + list(kumaMSD.parameters())
    optimizer       = optim.SGD(params,   lr=config['lr'])
    loss_function   = nn.CrossEntropyLoss()
    m_loss_function = nn.MSELoss()

    kl_weight       = config['kl_start']
    anneal_rate     = (1.0 - config['kl_start']) / (config['epochs'] * len(train_loader_sup))

    config['anneal_rate'] = anneal_rate

    print('-' * 13)
    print('    DEVICE')
    print('-' * 13)
    print(device)
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

    epoch_details = 'epoch, bce_loss, kl_div, kl_weight, loss\n'

    # init kuma prior
    a0 = torch.zeros(1, config['label_len']).to(device)
    b0 = torch.zeros(1, config['label_len']).to(device)
    kuma_prior = Kumaraswamy(a0, b0)

    model.train()
    kumaMSD.train()
    for epoch in range(config['epochs']):

        start      = timer()
        epoch_loss = 0

        it_sup      = iter(train_loader_sup)
        it_unsup    = iter(train_loader_unsup)
        done_sup    = False
        done_unsup  = False
        num_batches = 0

        # for i_batch, sample_batched in enumerate(train_loader):
        while True:
            if done_sup and done_unsup:
                break

            if done_sup:
                choice = 'unsup'
            elif done_unsup:
                choice = 'sup'
            else:
                choice = random.choice(['sup', 'unsup'])

            if choice == 'sup':
                try:
                    sample_batched = next(it_sup)
                except StopIteration:
                    done_sup = True
                    continue

            if choice == 'unsup':
                try:
                    sample_batched = next(it_unsup)
                except StopIteration:
                    done_unsup = True
                    continue

            # print("CHOICE = {}".format(choice))
            # print("****************")


            optimizer.zero_grad()

            x_s = sample_batched['source_form'].to(device)
            x_t = sample_batched['target_form'].to(device)
            y_t = sample_batched['msd']

            x_s = torch.transpose(x_s, 0, 1)
            x_t = torch.transpose(x_t, 0, 1)

            def process_unsup_msd(batch_y_t):
                y_t_len = batch_y_t.size(0)
                output  = []

                for i in range(y_t_len):
                    one_hot = [0.] * y_t_len
                    one_hot[i] = batch_y_t[i]
                    output.append(one_hot)

                return torch.tensor(output).to(device)

            y_t_p, kuma_post          = kumaMSD(x_t)

            # print(kuma_post.a)

            if choice == 'sup':
                y_t = y_t.to(device)
                y_t = torch.transpose(y_t, 0, 1)
                y_t_pp = y_t
            else:
                y_t_pp = torch.stack([process_unsup_msd(batch) for batch in y_t_p])
                y_t_pp = y_t_pp.permute(1, 0, 2).to(device)

            x_t_p, mu, logvar = model(x_s, x_t, y_t_pp)

            x_t_p = x_t_p[1:].view(-1, x_t_p.shape[-1])
            x_t_a = x_t[1:].contiguous().view(-1)

            bce_loss = loss_function(x_t_p, x_t_a)
            kl_term  = kl_div(mu, logvar)

            # ha bits , like free bits but over whole layer
            # REFERENCE: https://github.com/kastnerkyle/pytorch-text-vae
            habits_lambda  = config['lambda_m']
            clamp_KLD      = torch.clamp(kl_term.mean(), min=habits_lambda).squeeze()
            loss           = bce_loss + kl_weight * clamp_KLD

            kuma_kl        = loss_kuma(kuma_prior, kuma_post, supervised=(choice == 'sup'))
            clamp_kuma_kld = torch.clamp(kuma_kl.mean(), min=habits_lambda).squeeze()
            total_loss     = loss - clamp_kuma_kld

            # FIXME: This pushes the ai, bi values to Inf/NaN
            # if choice == 'sup':
            #     total_loss += torch.sum(kuma_post.log_prob(kuma_post.sample()))

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            num_batches   += 1
            epoch_loss    += loss.detach().cpu().item()
            epoch_details += '{}, {}, {}, {}, {}\n'.format(epoch,
                                                           bce_loss.detach().cpu().item(),
                                                           clamp_KLD.detach().cpu().item(),
                                                           kl_weight,
                                                           loss.detach().cpu().item())
            writer.add_scalar('BCE loss',   bce_loss.detach().cpu().item())
            writer.add_scalar('KLD',        clamp_KLD.detach().cpu().item())
            writer.add_scalar('kl_weight',  kl_weight)
            writer.add_scalar('Loss',       loss.detach().cpu().item())
            writer.add_scalar('Total Loss', total_loss.detach().cpu().item())
            #writer.add_scalar('M_Loss',    m_loss.detach().cpu().item())

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
                'optimizer_state_dict': optimizer.state_dict(),
                'loss'                : epoch_loss / (num_batches + 1),
                'config'              : config
            }, '../models/{}-{}/model.pt'.format(config['language'], config['model_id']))

    if dont_save:
        return

    with open('../models/{}-{}/epoch_details.csv'.format(config['language'], config['model_id']), 'w+') as f:
        f.write(epoch_details)


if __name__ == "__main__":

    parser                  = argparse.ArgumentParser()
    parser.add_argument('--train',       action='store_true')
    parser.add_argument('--test',        action='store_true')
    parser.add_argument('--dont_save',   action='store_true',        default=False)
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
    parser.add_argument('-a0',           action="store", type=float, default=0.0)
    parser.add_argument('-b0',           action="store", type=float, default=0.0)

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

    vocab                   = Vocabulary(language=config['language'])

    # TRAIN
    if run_train:
        train(config, vocab, dont_save)

    # TEST
    if run_test:
        test(config['language'], config['model_id'], vocab, dont_save)
