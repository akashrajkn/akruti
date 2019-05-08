from __future__ import print_function, division

import os
import torch

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from helper import get_label_length


class MorphologyDatasetTask3(Dataset):
    """Morphology reinflection dataset."""

    def __init__(self, csv_file, language, get_unprocessed=False, root_dir='../data/files', delimiter='\t'):
        """
        Args:
            csv_file (string) : Path to the csv file with annotations.
            language (string) : Language
            root_dir (string) : Directory with all the data.
        """
        self.csv_file        = csv_file
        self.language        = language
        self.root_dir        = root_dir
        self.get_unprocessed = get_unprocessed  # raw output

        self.pd_data     = pd.read_csv(csv_file, delimiter=delimiter, header=None)
        self.max_seq_len = self._max_sequence_length()

    def __len__(self):
        return len(self.pd_data)

    def __getitem__(self, idx):

        msd = self.pd_data.iloc[idx, 1]
        msds = msd.strip().split(',')

        sample = {
            'source_form': self._prepare_sequence(self.pd_data.iloc[idx, 0]),
            'msd'        : self._prepare_msd_each_feature(msds),
            'target_form': self._prepare_sequence(self.pd_data.iloc[idx, 2])
        }

        if self.get_unprocessed:
            sample['source_str'] = self.pd_data.iloc[idx, 0]
            sample['msd_str']    = self.pd_data.iloc[idx, 1]
            sample['target_str'] = self.pd_data.iloc[idx, 2]

        return sample

    def set_vocabulary(self, char_2_idx, idx_2_char, desc_2_idx, idx_2_desc, msd_types):
        # TODO: create a Vocabulary class later on
        self.char_2_idx  = char_2_idx
        self.idx_2_char  = idx_2_char
        self.desc_2_idx  = desc_2_idx
        self.idx_2_desc  = idx_2_desc
        self.msd_types   = msd_types
        self.padding_idx = char_2_idx['<PAD>']

        if msd_types is not None:
            self.label_len = get_label_length(idx_2_desc, msd_types) + 1  # last index is for None


    def get_vocab_size(self):
        return len(self.char_2_idx)

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
                idx = self.char_2_idx['<unk>']

            output.append(idx)

        output.append(self.char_2_idx['<EOS>'])

        while len(output) < self.max_seq_len:
            output.append(self.char_2_idx['<PAD>'])

        return torch.tensor(output).type(torch.LongTensor)

    def _prepare_msd_each_feature(self, msds):
        '''
        msds   -- ['pos=verb', 'tense=present', ...]
        output -- [[0, 1, 0, 0, ....] length: |msd_seq_len|]
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
