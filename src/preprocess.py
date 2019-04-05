import os
import pickle

from data import *

'''
    Vocabulary: unique characters
    dictionary: idx-to-char: save
    dictionary: idx-to-morp: save

    input seq : to list of idx
    morph feat: to list of idx
    output seq: to lsit of idx
'''

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

    print('Saving (Vocab) dictionaries')

    save_file('../data/pickles/idx_2_char', idx_2_char)
    save_file('../data/pickles/char_2_idx', char_2_idx)
    save_file('../data/pickles/idx_2_desc', idx_2_desc)
    save_file('../data/pickles/desc_2_idx', desc_2_idx)

    print('  - Done')


def main(filepath):

    all_out = read_task_3(filepath)

    if not os.path.exists('../data/pickles/idx_2_char'):
        convert_to_dicts(all_out)

    idx_2_char = load_file('../data/pickles/idx_2_char')
    char_2_idx = load_file('../data/pickles/char_2_idx')
    idx_2_desc = load_file('../data/pickles/idx_2_desc')
    desc_2_idx = load_file('../data/pickles/desc_2_idx')


if __name__ == '__main__':
    datapath = '../data/files/task3_test'
    main(datapath)
