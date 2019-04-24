from __future__ import print_function, division

import os
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MorphologyDatasetTask3(Dataset):
    """Morphology reinflection dataset."""

    def __init__(self, csv_file, language, root_dir='../data/files', delimiter='\t'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            language (string): Language
            root_dir (string): Directory with all the data.
        """
        self.csv_file = csv_file
        self.language = language
        self.root_dir = root_dir

        self.pd_data = pd.read_csv(csv_file, delimiter=delimiter, header=None)
        self.max_seq_len = self._max_sequence_length()

    def set_vocabulary(self, char_2_idx, idx_2_char, desc_2_idx, idx_2_desc, msd_types):
        # TODO: create a Vocabulary class later on
        self.char_2_idx  = char_2_idx
        self.idx_2_char  = idx_2_char
        self.desc_2_idx  = desc_2_idx
        self.idx_2_desc  = idx_2_desc
        self.msd_types   = msd_types
        self.padding_idx = char_2_idx['<PAD>']

    def __len__(self):
        return len(self.pd_data)

    def __getitem__(self, idx):

        msd = self.pd_data.iloc[idx, 1]
        out = {}
        msds = msd.strip().split(',')

        for m in msds:
            current = m.strip().split('=')
            out[current[0]] = current[1]

        sample = {
            'source_form': self._prepare_sequence(self.pd_data.iloc[idx, 0]),
            'msd'        : self._prepare_msd(out),
            'target_form': self._prepare_sequence(self.pd_data.iloc[idx, 2])
        }

        return sample

    def get_unprocessed_strings(self, idx):
        return {
            'source_form': self.pd_data.iloc[idx, 0],
            'msd'        : self.pd_data.iloc[idx, 1],
            'target_form': self.pd_data.iloc[idx, 2]
        }


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
        Append <EOS> to each sequence and Pad with <PAD>
        '''
        output = [self.char_2_idx['<SOS>']]

        for char in sequence:
            output.append(self.char_2_idx[char])

        output.append(self.char_2_idx['<EOS>'])

        while len(output) < self.max_seq_len:
            output.append(self.char_2_idx['<PAD>'])

        return torch.tensor(output).type(torch.LongTensor)

    def _prepare_msd(self, msd):
        '''
        msd   : {'pos': 'verb', 'tense': 'present', 'mod': 'ind'}
        output: [0, 5, 7, 10, ...] length: |label_types|
        '''
        label_types = len(self.idx_2_desc)
        output  = []

        for i in range(label_types):
            desc  = self.idx_2_desc[i]
            opt   = msd.get(desc)
            types = self.msd_types[i]

            if opt is None:
                opt = 'None'

            output.append(types[opt])

        return torch.tensor(output).type(torch.LongTensor)
