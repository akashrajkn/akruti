from __future__ import print_function, division

import os
import torch

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from helper import get_label_length


class Vocabulary():
    """Vocabulary"""
    def __init__(self, language, path=None):

        self.language   = language
        self.char_2_idx = {}
        self.idx_2_char = {}
        self.vocab_size = 0

        self.desc_2_idx = {}
        self.idx_2_desc = {}
        self.msd_size   = 0

        for c in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']:
            self.char_2_idx[c]               = self.vocab_size
            self.idx_2_char[self.vocab_size] = c
            self.vocab_size                 += 1

        self.desc_2_idx['<unkMSD>']    = self.msd_size
        self.idx_2_desc[self.msd_size] = '<unkMSD>'
        self.msd_size                 += 1

        # Read files
        common_path   = '../data/files/{}'.format(language)
        tasks         = ['task1p', 'task2p', 'task3']
        f_types       = ['dev', 'test', 'train']

        for task in tasks:
            for f_type in f_types:
                filepath = common_path + '-{}-{}'.format(task, f_type)

                if os.path.isfile(filepath):
                    self._preprocess(filepath)

    def _process_word(self, word):
        for character in word:
            if self.char_2_idx.get(character) is None:
                self.char_2_idx[character]       = self.vocab_size
                self.idx_2_char[self.vocab_size] = character
                self.vocab_size                 += 1

    def _process_msds(self, msd_line):

        if msd_line == '<UNLABELED>':
            return

        msds = msd_line.strip().split(',')

        for msd in msds:
            if self.desc_2_idx.get(msd) is None:
                self.desc_2_idx[msd]           = self.msd_size
                self.idx_2_desc[self.msd_size] = msd
                self.msd_size                 += 1

    def _preprocess(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        out       = []
        sentences = source.strip().split('\n')

        for sentence in sentences:
            line  = sentence.strip().split('\t')

            if len(line) > 3:
                print('Something wrong with line: {}'.format(sentence))
                continue

            self._process_word(line[0])
            self._process_msds(line[1])
            self._process_word(line[2])


class MorphologyDatasetTask3(Dataset):
    """Morphology reinflection dataset."""

    def __init__(self, test, language, vocab, tasks, get_unprocessed=False, delimiter='\t'):
        """
        Args:
            test (string)     : train or test
            language (string) : Language
            tasks (list)      : ['task1p', 'task2p']
        """
        self.test            = test
        self.language        = language
        self.tasks           = tasks
        self.get_unprocessed = get_unprocessed  # raw output
        self.delimiter       = delimiter

        self._get_pd_data()
        self.max_seq_len = self._max_sequence_length()

        # Set Vocabulary
        self.char_2_idx  = vocab.char_2_idx
        self.idx_2_char  = vocab.idx_2_char
        self.desc_2_idx  = vocab.desc_2_idx
        self.idx_2_desc  = vocab.idx_2_desc
        self.padding_idx = vocab.char_2_idx['<PAD>']

    def __len__(self):
        return len(self.pd_data)

    def __getitem__(self, idx):

        msd  = self.pd_data.iloc[idx, 1]
        msds = '<UNLABELED>'

        if msd != '<UNLABELED>':
            msds = self._prepare_msd_each_feature(msd.strip().split(','))

        sample = {
            'source_form': self._prepare_sequence(self.pd_data.iloc[idx, 0]),
            'msd'        : msds,
            'target_form': self._prepare_sequence(self.pd_data.iloc[idx, 2])
        }

        if self.get_unprocessed:
            sample['source_str'] = self.pd_data.iloc[idx, 0]
            sample['msd_str']    = self.pd_data.iloc[idx, 1]
            sample['target_str'] = self.pd_data.iloc[idx, 2]

        return sample

    def get_vocab_size(self):
        return len(self.char_2_idx)

    def _get_pd_data(self):
        common_path   = '../data/files/{}'.format(self.language)

        if self.test:
            self.pd_data  = pd.read_csv(common_path + '-task3-test', delimiter=self.delimiter, header=None)
            return

        if self.tasks[0] == 'task3':
            self.pd_data  = pd.read_csv(common_path + '-task3-train', delimiter=self.delimiter, header=None)
            return

        f_types       = ['train', 'test', 'dev']
        frames        = []

        for task in self.tasks:
            for f_type in f_types:

                if task == 'task1p' and f_type == 'dev':
                    continue

                filepath = common_path + '-{}-{}'.format(task, f_type)

                if os.path.isfile(filepath):
                    data = pd.read_csv(filepath, delimiter=self.delimiter, header=None)
                    frames.append(data)

        self.pd_data = pd.concat(frames)

    def _max_sequence_length(self):
        '''
        Return the length of the longest source/target sequence
        '''
        max_len = 0

        for i in range(len(self.pd_data)):
            source_len = len(self.pd_data.iloc[i, 0])
            target_len = len(self.pd_data.iloc[i, 2])

            if (source_len > max_len):
                max_len = source_len

            if (target_len > max_len):
                max_len = target_len

        return max_len + 2  # + 2 is for <SOS> & <EOS> char

    def _prepare_sequence(self, sequence):
        '''
        - Append <EOS> to each sequence and Pad with <PAD>
        - If test set contains characters that didn't occur in the train set,
            <unk> is used
        '''
        output = [self.char_2_idx['<SOS>']]

        for char in sequence:
            idx = self.char_2_idx.get(char)

            if idx is None:
                idx = self.char_2_idx['<UNK>']

            output.append(idx)

        output.append(self.char_2_idx['<EOS>'])

        while len(output) < self.max_seq_len:
            output.append(self.char_2_idx['<PAD>'])

        return torch.tensor(output).type(torch.LongTensor)

    def _prepare_msd_each_feature(self, msds):
        '''
        msds   -- ['pos=verb', 'tense=present', ...]
        output -- [[0, 1, 0, 0, ....], [...]] length: |msd_seq_len| * |msd_seq_len|
        '''
        msd_seq_len = len(self.desc_2_idx)
        output      = []

        for idx in self.idx_2_desc:
            one_hot = [0] * msd_seq_len

            # NOTE, FIXME: This does not take into account <unkMSD>.
            if self.idx_2_desc[idx] in msds:
                one_hot[idx] = 1

            output.append(one_hot)

        # for m in msds:
        #     one_hot = [0] * msd_seq_len
        #     idx     = self.desc_2_idx.get(m)
        #
        #     if idx is None:
        #         idx = self.desc_2_idx['<unkMSD>']
        #
        #     one_hot[idx] = 1
        #     output.append(one_hot)
        #
        # while len(output) < msd_seq_len:
        #     output.append([0] * msd_seq_len)  # equivalent of giving padding_idx to nn.Embedding

        return torch.tensor(output).type(torch.FloatTensor)


    def _prepare_msd(self, msds):
        '''
        msds   -- {'pos': 'verb', 'tense': 'present', 'mod': 'ind'}
        output -- [0, 1, 0, 0, ....] length: |label_len|
        # output -- [0, 5, 7, 10, ...] length: |label_types|
        '''
        msd = {}

        for m in msds:
            current = m.strip().split('=')
            msd[current[0]] = current[1]

        label_types = len(self.idx_2_desc)
        output      = []

        for i in range(label_types):
            one_hot = [0] * self.label_len

            desc  = self.idx_2_desc[i]
            opt   = msd.get(desc)
            types = self.msd_types[i]

            # if opt is None:
            #     one_hot[self.label_len - 1] = 1
            # else:
            #     one_hot[types[opt]] = 1

            # FIXME: This could be dangerous - check it out again
            try:
                one_hot[types[opt]] = 1
            except:
                one_hot[self.label_len - 1] = 1

            output.append(one_hot)

            # TODO: Return vector of zeros instead of padding index - because I am using torch.bmm

        return torch.tensor(output).type(torch.FloatTensor)
