import os
import pickle
import sys

from helper import *


def convert_to_dicts(all_out):
    '''
    (Task 3)
    Find vocabulary, generate dictionary and save files
    Args:
        all_out: list of dictionaries. each dictionary contains source, msd, target
    '''
    source_forms = []
    msds         = []
    target_forms = []

    for triplet in all_out:
        source_forms.append(triplet['source_form'])
        msds.append(triplet['MSD'])
        target_forms.append(triplet['target_form'])

    all_words  = source_forms + target_forms
    vocab_size = 0
    char_2_idx = {}
    idx_2_char = {}

    char_2_idx['<SOS>']    = vocab_size
    idx_2_char[vocab_size] = '<SOS>'
    vocab_size            += 1

    char_2_idx['<EOS>']    = vocab_size
    idx_2_char[vocab_size] = '<EOS>'
    vocab_size            += 1

    char_2_idx['<PAD>']    = vocab_size
    idx_2_char[vocab_size] = '<PAD>'
    vocab_size            += 1

    for word in all_words:
        for character in word:
            if char_2_idx.get(character) is None:
                char_2_idx[character] = vocab_size
                idx_2_char[vocab_size] = character
                vocab_size += 1

    msd_size   = 0
    desc_2_idx = {}
    idx_2_desc = {}

    for obj in msds:
        for key in obj.keys():
            if desc_2_idx.get(key) is None:
                desc_2_idx[key] = msd_size
                idx_2_desc[msd_size] = key
                msd_size += 1

    # MSD options dict
    msd_options = {}

    count = 0

    for key, value in desc_2_idx.items():
        current_options = {"None": count}
        count += 1
        for msd in msds:
            for k, v in msd.items():
                if k == key:
                    if current_options.get(v) is None:
                        current_options[v] = count
                        count += 1

        msd_options[value] = current_options

    print('Saving (Vocab) dictionaries')

    save_file('../data/pickles/idx_2_char',  idx_2_char)
    save_file('../data/pickles/char_2_idx',  char_2_idx)
    save_file('../data/pickles/idx_2_desc',  idx_2_desc)
    save_file('../data/pickles/desc_2_idx',  desc_2_idx)
    save_file('../data/pickles/msd_options', msd_options)

    print('  - Done')


def main(filepath, rewrite=False):

    all_out = read_task_3(filepath)

    if (not os.path.exists('../data/pickles/idx_2_char')) or rewrite:
        convert_to_dicts(all_out)
    else:
        print('Dictionaries already exist. Re-run')


if __name__ == '__main__':
    datapath = '../data/files/turkish-task3-train'

    rewrite = False

    if len(sys.argv) <= 1:
        rewrite = False
    elif sys.argv[1] == 'rewrite':
        rewrite = True

    main(datapath, rewrite=rewrite)
