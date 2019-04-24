import logging
import argparse
import numpy as np

from torch.utils.data import DataLoader
from timeit import default_timer as timer
from datetime import timedelta

from models import *
from helper import load_file, get_label_length
from dataset import MorphologyDatasetTask3


def kl_div(mu, logvar):
    '''
    Compute KL divergence between N(mu, logvar) and N(0, 1)
    '''
    return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def test(model, test_morph_data, config, idx_2_char):
    '''
    Test function
    TODO implement accuracy
    '''
    print('-' * 13)
    print('    TEST')
    print('-' * 13)

    device = config['device']

    for i in range(len(test_morph_data)):

        if i == 10:
            break

        with torch.no_grad():
            sample = test_morph_data[i]

            x_s = sample['source_form'].to(device)
            x_s = torch.unsqueeze(x_s, 1)
            x_s = torch.transpose(x_s, 0, 1)

            y_t = sample['msd'].to(device)
            y_t = torch.unsqueeze(y_t, 1)
            y_t = torch.transpose(y_t, 0, 1)

            x_t_p, _, _ = model(x_s, y_t)
            x_t_p       = x_t_p[1:].view(-1, x_t_p.shape[-1])

            outputs     = F.log_softmax(x_t_p, dim=1).type(torch.LongTensor)
            outputs     = torch.squeeze(outputs, 1)

            target_word = ''
            for i in outputs:
                p            = np.argmax(i, axis=0).detach().cpu().item()
                target_word += idx_2_char[p]

            print('Target   : {}'.format(test_morph_data.get_unprocessed_strings(i)))
            print('Predicted: {}'.format(target_word))


def train(train_dataloader, config, model_file):
    '''
    Train function
    '''
    device        = config['device']
    # Model declaration
    encoder       = WordEncoder(config['vocab_size'])   # TODO: give padding_idx
    tag_embedding = TagEmbedding(config['label_len'])
    attention     = Attention()
    decoder       = WordDecoder(attention, config['vocab_size'])
    model         = MSVED(encoder, tag_embedding, decoder, config['max_seq_len'], device).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer     = optim.SGD(model.parameters(), lr=0.1)

    print('-' * 13)
    print('    DEVICE =', device)
    print('-' * 13)
    print('    MODEL  :')
    print(model)
    print('-' * 13)
    print('    TRAIN  :')
    print('-' * 13)

    model.train()
    for epoch in range(config['epochs']):

        start      = timer()
        epoch_loss = 0
        count      = 0

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
            x_t   = x_t[1:].view(-1)

            loss = loss_function(x_t_p, x_t) + config['lambda_m'] * kl_div(mu, logvar)
            loss.backward()
            optimizer.step()

            current_loss = loss.detach().cpu().item()
            epoch_loss  += current_loss
            count       += 1
            break

        end = timer()
        break

        print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss / count))
        print('         - Time: {}'.format(timedelta(seconds=end - start)))

        print('         - Save model')
        torch.save(model, '../models/model-{}-epoch_{}.pt'.format(model_file, str(epoch)))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Set up data
    idx_2_char = load_file('../data/pickles/idx_2_char')
    char_2_idx = load_file('../data/pickles/char_2_idx')
    idx_2_desc = load_file('../data/pickles/idx_2_desc')
    desc_2_idx = load_file('../data/pickles/desc_2_idx')
    msd_types  = load_file('../data/pickles/msd_options')  # label types

    config = {}
    config['epochs']        = 5
    config['h_dim']         = 256
    config['z_dim']         = 150
    config['lambda_m']      = 0.2  # TODO: linear/exp anneal term
    config['bidirectional'] = True
    config['batch_size']    = 1
    config['device']        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['vocab_size']    = len(char_2_idx)
    config['label_len']     = get_label_length(idx_2_desc, msd_types)  # TODO: move this to Dataset class

    # Get train_dataloader
    train_file       = 'turkish-task3-test'
    language         = 'turkish'
    morph_data       = MorphologyDatasetTask3(csv_file='../data/files/{}.csv'.format(train_file), language=language)
    morph_data.set_vocabulary(char_2_idx, idx_2_char, desc_2_idx, idx_2_desc, msd_types)
    train_dataloader = DataLoader(morph_data, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    config['max_seq_len'] = morph_data.max_seq_len

    # TRAIN
    model = train(train_dataloader, config, model_file=train_file)

    # Get test_dataloader
    test_file             = 'turkish-task3-test'
    test_morph_data       = MorphologyDatasetTask3(csv_file='../data/files/{}.csv'.format(test_file), language=language)
    test_morph_data.set_vocabulary(char_2_idx, idx_2_char, desc_2_idx, idx_2_desc, msd_types)

    # TEST
    test(model, test_morph_data, config, idx_2_char)
