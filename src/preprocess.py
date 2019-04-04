import os

from data import *

'''
    Vocabulary: unique characters
    dictionary: idx-to-char: save
    dictionary: idx-to-morp: save

    input seq : to list of idx
    morph feat: to list of idx
    output seq: to lsit of idx
'''

def main(filepath):

    all_out = read_task_3(filepath)

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

    print(char_2_idx)
    print(idx_2_char)


if __name__ == '__main__':

    path = '../data/files/task3_test'
    main(path)
