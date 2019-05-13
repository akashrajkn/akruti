import argparse
import os
import pickle
import sys

import helper


def get_msd_dict_each_feature(msds):
    '''
    Converts msds to dictionary where each msd option is a different feature
    '''
    msd_size   = 0
    desc_2_idx = {}
    idx_2_desc = {}


    for msd in msds:
        for key in msd:
            if desc_2_idx.get(key) is None:
                desc_2_idx[key]      = msd_size
                idx_2_desc[msd_size] = key
                msd_size            += 1

    desc_2_idx['<unkMSD>'] = msd_size
    idx_2_desc[msd_size]   = '<unkMSD>'
    msd_size              += 1

    return desc_2_idx, idx_2_desc, None

def get_msd_dict_with_types(msds):
    ''''
    Converts msds to dictionary
    '''
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
    count       = 0

    for key, value in desc_2_idx.items():
        current_options = {}
        for msd in msds:
            for k, v in msd.items():
                if k == key:
                    if current_options.get(v) is None:
                        current_options[v] = count
                        count += 1

        msd_options[value] = current_options

    return desc_2_idx, idx_2_desc, msd_options

def convert_to_dicts(all_out, language):
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

    char_2_idx['<unk>']    = vocab_size
    idx_2_char[vocab_size] = '<unk>'
    vocab_size            += 1

    for word in all_words:
        for character in word:
            if char_2_idx.get(character) is None:
                char_2_idx[character] = vocab_size
                idx_2_char[vocab_size] = character
                vocab_size += 1

    desc_2_idx, idx_2_desc, msd_options = get_msd_dict_each_feature(msds)

    print('Saving (Vocab) dictionaries')

    helper.save_file('../data/pickles/{}-idx_2_char'.format(language), idx_2_char)
    helper.save_file('../data/pickles/{}-char_2_idx'.format(language), char_2_idx)
    helper.save_file('../data/pickles/{}-idx_2_desc'.format(language), idx_2_desc)
    helper.save_file('../data/pickles/{}-desc_2_idx'.format(language), desc_2_idx)

    if msd_options is not None:
        helper.save_file('../data/pickles/{}-msd_options'.format(language), msd_options)

    print('  - Done')


def main(language, rewrite=False):

    datapath = '../data/files/{}-task3-train'.format(language)
    all_out  = helper.read_task_3(datapath)

    if (not os.path.exists('../data/pickles/{}-idx_2_char'.format(language))) or rewrite:
        convert_to_dicts(all_out, language)
    else:
        print('Dictionaries already exist. Re-run')


def process_task2_files(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    out       = []
    sentences = source.strip().split('\n')

    for sentence in sentences:
        line  = sentence.strip().split('\t')
        line  = line[1:4]
        out.append('\t'.join(line))

    with open(filepath.replace('task2', 'task2p'), 'w+') as f:
        f.write('\n'.join(out))

def modify_task2_files():

    common_path = '../data/files/'

    for lang in ['arabic', 'finnish', 'georgian', 'german', 'hungarian', 'maltese', 'navajo', 'russian', 'spanish', 'turkish']:

        n_p = common_path + '{}-task2-'.format(lang)
        for t in ['train', 'test', 'dev']:
            filepath = n_p + t
            process_task2_files(filepath)


if __name__ == '__main__':

    parser   = argparse.ArgumentParser()
    parser.add_argument('--rewrite', action='store_true')
    parser.add_argument('-language', action="store", type=str)

    args     = parser.parse_args()
    rewrite  = args.rewrite
    language = args.language

    # main(language=language, rewrite=rewrite)
    modify_task2_files()
