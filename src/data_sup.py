# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict
import codecs
import sys
import logging
import collections

logger = logging.getLogger(__name__)
UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)
'''Task 1
For task 1, the fields are: lemma, MSD, target form. An example from the Spanish training data:
hablar  pos=V,mood=IND,polite=FORM,tense=FUT,per=3,num=SG       hablara
Task 2
In task 2, the fields are: source MSD, source form, target MSD, target form. For example:
pos=V,mood=IND,tense=PRS,per=1,num=SG,aspect=IPFV/PFV   hablo   pos=V,tense=PST,gen=MASC,num=PL hablados
Task 3
In task 3, the fields are: source form, target MSD, target form. For example:
hablo   pos=V,tense=PST,gen=MASC,num=PL hablados'''

datapath = '../data/'
langs = ['spanish', 'german', 'finnish', 'russian', 'turkish', 'georgian', 'navajo', 'arabic', 'hungarian', 'maltese']
task1_train = '-task1-train'
task1_test = '-task1-test' # treat as unlabeled data first
task1_dev = '-task1-dev' # test data
task2_train = '-task2-train'
task2_test = '-task2-test' # treat as unlabeled data first
task2_dev = '-task2-dev'
task3_train = '-task3-train'
task3_test = '-task3-test' # treat as unlabeled data first
task3_dev = '-task3-dev'

tags_pre = {}

# dataset is small, we load it all at once
ux = [] # each item is a list of char idx
uy = [] # each item is a list of label idx for all tags; convert to 1-of-k later
lx_src = []
ly_src = []
lx_tgt = []
ly_tgt = []
l_set = dict()
unique_data_set_train = set()
unlabeled_words = set()
lang = ""

def read_task1(fname, allwords):
    # lemma target_label target word
    with codecs.open(fname, 'r', "utf-8") as f:
        for line in f:
            fields = line.strip().split('\t')
            tags = fields[1].split(',')
            word_1 = fields[0]
            word_2 = fields[2]
            allwords += word_1 + word_2

            word = word_1 + word_2
            if '{' in word or '}' in word:
                logging.info(word)

            data_pair = ' '.join([fields[1], word_2])
            unique_data_set_train.add(data_pair)
            unlabeled_words.add(word_2)
            for tag in tags:
                att, value = tag.split('=')
                if att in tags_pre:
                    if value not in tags_pre[att]:
                        tags_pre[att].add(value)
                else:
                    tags_pre[att] = set([value])
    return allwords


def read_task2(fname, allwords, test=True):
    # source_label source_form target_label target_form
    with codecs.open(fname, 'r', "utf-8") as f:
        for line in f:
            fields = line.strip().split('\t')
            tags = fields[0].split(',') + fields[2].split(',')
            allwords = allwords + fields[1] + fields[3]
            if test:
                unique_data_set_train.add(' '.join([fields[0], fields[1]]))
                unique_data_set_train.add(' '.join([fields[2], fields[3]]))
                unlabeled_words.add(fields[1])
                unlabeled_words.add(fields[3])
            for tag in tags:
                att, value = tag.split('=')
                if att in tags_pre:
                    if value not in tags_pre[att]:
                        tags_pre[att].add(value)
                else:
                    tags_pre[att] = set([value])
    return allwords


def read_task3(fname, allwords):
    # source_form target_label target_form
    with codecs.open(fname, 'r', "utf-8") as f:
        for line in f:
            fields = line.strip().split('\t')
            tags = fields[1].split(',')
            word_1 = fields[0]
            word_2 = fields[2]
            allwords += word_1 + word_2

            word = word_1 + word_2
            if '{' in word or '}' in word:
                logging.info(word)

            data_pair = ' '.join([fields[1], word_2])
            # unique_data_set_train.add(data_pair)
            for tag in tags:
                att, value = tag.split('=')
                if att in tags_pre:
                    if value not in tags_pre[att]:
                        tags_pre[att].add(value)
                else:
                    tags_pre[att] = set([value])
    return allwords


def create_test_data(fname, char_to_ix, label_to_ix, tag_to_ix, class_num):
    # task3 test: source_form target_label target_form
    x_src = []
    x_tgt = []
    y_tgt = []
    with codecs.open(fname, 'r', "utf-8") as f:
        for line in f:
            fields = line.strip().split('\t')
            srcx = fields[0]
            tgtx = fields[2]
            tgty = fields[1].split(',')
            word = []
            for char in srcx:
                word.append(char_to_ix[char])
            x_src.append(word)
            labels = [-1]*class_num
            word = []
            for tag in tgty:
                tag, label = tag.split('=')
                tag_id = tag_to_ix[tag]
                labels[tag_id] = label_to_ix[tag_id][label]
            labels = [label if label != -1 else label_to_ix[i]['None'] for i, label in enumerate(labels)]
            for char in tgtx:
                word.append(char_to_ix[char])
            x_tgt.append(word)
            y_tgt.append(labels)
    return x_src, x_tgt, y_tgt


def create_train_sup_task2(fname, char_to_ix, label_to_ix, tag_to_ix, class_num):
    with codecs.open(fname, 'r', "utf-8") as f:
        for line in f:
            fields = line.strip().split('\t')
            srcx = fields[1]
            srcy = fields[0].split(',')
            tgtx = fields[3]
            tgty = fields[2].split(',')
            src_labels = [-1]*class_num
            src_word = []
            for tag in srcy:
                tag, label = tag.split('=')
                tag_id = tag_to_ix[tag]
                src_labels[tag_id] = label_to_ix[tag_id][label]
            src_labels = [label if label != -1 else label_to_ix[i]['None'] for i, label in enumerate(src_labels)]
            for char in srcx:
                src_word.append(char_to_ix[char])
            # lx_src.append(src_word)
            # ly_src.append(src_labels)
            tgt_labels = [-1]*class_num
            tgt_word = []
            for tag in tgty:
                tag, label = tag.split('=')
                tag_id = tag_to_ix[tag]
                tgt_labels[tag_id] = label_to_ix[tag_id][label]
            tgt_labels = [label if label != -1 else label_to_ix[i]['None'] for i, label in enumerate(tgt_labels)]
            for char in tgtx:
                tgt_word.append(char_to_ix[char])
            # lx_tgt.append(tgt_word)
            # ly_tgt.append(tgt_labels)
            key_1 = src_word + ['*'] + tgt_word
            key_2 = tgt_word + ['*'] + src_word
            key_1 = [str(i) for i in key_1]
            key_2 = [str(i) for i in key_2]
            key_1 = ''.join(key_1)
            key_2 = ''.join(key_2)

            data_1 = [src_word] + [src_labels] + [tgt_word] + [tgt_labels]
            data_2 = [tgt_word] + [tgt_labels] + [src_word] + [src_labels]
            if key_1 not in l_set:
                l_set[key_1] = data_1
            if key_2 not in l_set:
                l_set[key_2] = data_2
    for item in l_set.values():
        lx_src.append(item[0])
        ly_src.append(item[1])
        lx_tgt.append(item[2])
        ly_tgt.append(item[3])


def create_train_sup_task3(fname, char_to_ix, label_to_ix, tag_to_ix, class_num):
    # task3 test: source_form target_label target_form
    with codecs.open(fname, 'r', "utf-8") as f:
        for line in f:
            fields = line.strip().split('\t')
            srcx = fields[0]
            tgtx = fields[2]
            tgty = fields[1].split(',')
            word = []
            for char in srcx:
                word.append(char_to_ix[char])
            lx_src.append(word)
            labels = [-1]*class_num
            word = []
            for tag in tgty:
                tag, label = tag.split('=')
                tag_id = tag_to_ix[tag]
                labels[tag_id] = label_to_ix[tag_id][label]
            labels = [label if label != -1 else label_to_ix[i]['None'] for i, label in enumerate(labels)]
            for char in tgtx:
                word.append(char_to_ix[char])
            lx_tgt.append(word)
            ly_tgt.append(labels)


def create_train_data(char_to_ix, label_to_idx, tag_to_ix, class_num):
    logging.info(tag_to_ix)
    for data in unique_data_set_train:
        fields = data.split(' ')
        tags = fields[0].split(',')
        word = fields[1]
        labels = [-1]*class_num
        chars = []
        for tag in tags:
            tag, label = tag.split('=')
            tag_id = tag_to_ix[tag]
            labels[tag_id] = label_to_idx[tag_id][label]
        labels = [label if label != -1 else label_to_idx[i]['None'] for i, label in enumerate(labels)]
        for char in word:
            chars.append(char_to_ix[char])
        ux.append(chars)
        uy.append(labels)


def create_unlabeled_data(char_to_ix, tot, fname):
    with codecs.open(fname, "r", "utf-8") as fin:
        i = 0
        added = 0
        for line in fin:
            if i < tot:
                word = line.strip()
                if word not in unlabeled_words:
                    w = [char_to_ix[j] for j in word]
                    added += 1
                    ux.append(w)
            i += 1
    logging.info("Add additional unlabeled data: %d", added)


def create_tag_dict():
    ix_to_tag = OrderedDict({i: tag for i, tag in enumerate(tags_pre)})
    tag_to_ix = OrderedDict({tag: i for i, tag in enumerate(tags_pre)})
    label_to_ix = []
    ix_to_label = []
    for i, tag in enumerate(tags_pre):
        temp_ix_to_label = OrderedDict({i+1: label for i, label in enumerate(tags_pre[tag])})
        temp_ix_to_label[0] = 'None'
        temp_label_to_ix = OrderedDict({label: i+1 for i, label in enumerate(tags_pre[tag])})
        temp_label_to_ix['None'] = 0
        label_to_ix.append(temp_label_to_ix)
        ix_to_label.append(temp_ix_to_label)
    return label_to_ix, ix_to_label, tag_to_ix, ix_to_tag


def preprocess(addsup=0.0):
    # task1_train_data = datapath + lang + task1_train
    # task1_test_data = datapath + lang + task1_test
    # task1_dev_data = datapath + lang + task1_dev
    # task2_train_data = datapath + lang + task2_train
    # task2_test_data = datapath + lang + task2_test
    # task2_dev_data = datapath + lang + task2_dev
    # task3_train_data = datapath + lang + task3_train
    # task3_test_data = datapath + lang + task3_test

    task3_train_data = '../data/files/task3_test'
    task3_test_data  = '../data/files/task1_test'

    # test_data = datapath + lang + task2_dev
    allwords = ""

    # allwords = read_task3(task3_train_data, allwords)
    # allwords = read_task1(task1_train_data, allwords)
    # allwords = read_task1(task1_test_data, allwords)
    # allwords = read_task2(task2_train_data, allwords, test=True)
    # allwords = read_task2(task2_test_data, allwords, test=True)
    # allwords = read_task2(task2_dev_data, allwords, test=True)

    allwords = read_task1('../data/files/task1_test', allwords)
    allwords = read_task3('../data/files/task3_test', allwords)
    allwords = read_task2('../data/files/task2_test', allwords, test=True)

    chars = list(set(allwords))
    data_size, vocab_size = len(allwords), len(chars)
    logging.info('data has %d characters, %d unique.' % (data_size, vocab_size))
    char_to_ix = {ch: i+1 for i, ch in enumerate(chars)}
    ix_to_char = {i+1: ch for i, ch in enumerate(chars)}

    class_num = len(tags_pre)
    label_list = [len(v)+1 for v in tags_pre.values()]
    voc_size = len(char_to_ix) + 1

    label_to_ix, ix_to_label, tag_to_ix, ix_to_tag = create_tag_dict()
    # unlabeled data: task1_train, task1_test, task3_train, task3_test, task2_train
    create_train_data(char_to_ix, label_to_ix, tag_to_ix, class_num)
    # # labeled data: task2_train
    # create_train_sup_task2(task2_labeled, char_to_ix, label_to_ix, tag_to_ix, class_num)
    # labeled data: task3_train
    create_train_sup_task3(task3_train_data, char_to_ix, label_to_ix, tag_to_ix, class_num)
    # test data: task3_test
    x_test_src, x_test_tgt, y_test_tgt = create_test_data(task3_test_data, char_to_ix, label_to_ix, tag_to_ix, class_num)

    if addsup > 0.0:
        tot = 180000
        if addsup < 1.0:
            tot = tot * addsup
        logging.info("Get additional unlabel data!")
        create_unlabeled_data(char_to_ix, tot, "../wiki/"+lang+"_wiki_voc")

    return voc_size, class_num, label_list, ix_to_char, ix_to_label, x_test_src, x_test_tgt, y_test_tgt


def get_batches(data_length, batch_size):
    num_batch = int(np.ceil(data_length/float(batch_size)))
    return [(i*batch_size, min(data_length, (i+1)*batch_size)) for i in range(0, num_batch)]


def prepare_xy_batch(seqs_x, seqs_y, label_list, maxlen=None):
    # x: a list of words
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        new_seqs_y = []
        for l_x, s_x, s_y in zip(lengths_x, seqs_x, seqs_y):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
            else:
                logging.info("length, x; ", l_x, s_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        seqs_y = new_seqs_y

        if len(lengths_x) < 1:
            return None, None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1

    x = np.zeros((n_samples, maxlen_x)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen_x)).astype(theano.config.floatX)

    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx]+1] = 1.

    y = []
    seqs_y = np.array(seqs_y, dtype='int32')
    for i in range(0, len(label_list)):
        _y = np.zeros((n_samples, label_list[i])).astype('float32')
        _y[range(n_samples), seqs_y[:, i]] = 1.
        y.append(_y)

    return x, x_mask, y


def prepare_x_batch(seqs_x, maxlen=None):
    # x: a list of words
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
            else:
                logging.info("length, x; ", l_x, s_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1

    x = np.zeros((n_samples, maxlen_x)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen_x)).astype(theano.config.floatX)

    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx] + 1] = 1.

    return x, x_mask

if __name__ == '__main__':
    # test
    if False:
        voc_size, class_num, label_list, ix_to_char, ix_to_label = preprocess()
        logging.info("number of unlabeled training set: ", len(ux))
        logging.info(voc_size, class_num, label_list, ix_to_char, ix_to_label)

    if True:
        f1 = "task1_test"
        f2 = "task2_test"

        allwords = ""
        allwords = read_task1(f1, allwords)
        allwords = read_task2(f2, allwords)

        logging.info(tags_pre)
        chars = list(set(allwords))
        data_size, vocab_size = len(allwords), len(chars)
        logging.info('data has %d characters, %d unique.' % (data_size, vocab_size))
        char_to_ix = OrderedDict({ch: i+1 for i, ch in enumerate(chars)})
        ix_to_char = OrderedDict({i+1: ch for i, ch in enumerate(chars)})

        class_num = len(tags_pre)
        label_list = [len(v)+1 for v in tags_pre.values()]
        voc_size = len(char_to_ix) + 1

        label_to_ix, ix_to_label, tag_to_ix, ix_to_tag = create_tag_dict()
        create_train_data(char_to_ix, label_to_ix, tag_to_ix, class_num)

        logging.info("char to ix: ", char_to_ix)
        logging.info("ix to char: ", ix_to_char)
        logging.info("tag to ix: ", tag_to_ix)
        logging.info("ix to tag: ", ix_to_tag)
        logging.info("label to ix: ", label_to_ix)
        logging.info("ix to label: ", ix_to_label)
        logging.info("label list: ", label_list)
        f_1 = open(f1)
        f_2 = open(f2)
        # logging.info(f_1.read())
        logging.info(f_2.read())
        f_1.close()
        f_2.close()

        create_train_sup_task2(f2, char_to_ix, label_to_ix, tag_to_ix, class_num)
        logging.info("srcx: ", lx_src)
        logging.info("srcy: ", ly_src)
        logging.info("tgtx: ", lx_tgt)
        logging.info("tgty: ", ly_tgt)
        # logging.info("train x: ", x)
        # logging.info("train y: ", y)
        #
        # x, x_mask, y = prepare_xy_batch(x[:3], y[:3], label_list, 6)
        # logging.info(x)
        # logging.info(x_mask)
        # logging.info(y)
