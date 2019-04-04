import sys
import os
import pickle
import logging
import argparse
import codecs
import time
import numpy as np

import data_sup


def main(config):
    voc_size, class_num, label_list, ix_to_char, ix_to_label, x_test_src, x_test_tgt, y_test_tgt = data_sup.preprocess(config["add_uns"])

    print(voc_size)
    print(class_num)
    print(label_list)
    print(ix_to_char)
    print(ix_to_label)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bidirectional", action="store_true", default=False)
    parser.add_argument("-y_dim", action="store", type=int, default=200)
    parser.add_argument("-z_dim", action="store", type=int, default=100)
    parser.add_argument("-cross_only", action="store_true", default=False)
    parser.add_argument("-disable_kl", action="store_true", default=False)
    parser.add_argument("-worddrop", action="store", type=float, default=0.3)
    parser.add_argument("-share_emb", action="store_false", default=True)
    parser.add_argument("-lang", action="store", type=str, default="arabic")
    parser.add_argument("-addsup", action="store_true", default=False)
    parser.add_argument("-kl_st", action="store_true", default=False)
    parser.add_argument("-kl_rate", action="store", type=float, default=150000.0)
    parser.add_argument("-kl_thres", action="store", type=float, default=0.4)
    parser.add_argument("-start_val", action="store", type=int, default=24000)
    parser.add_argument("-only_supervise", action="store_true", default=False)
    parser.add_argument("-dt_uns", action="store", type=float, default=0.7)
    parser.add_argument("-epochs", action="store", type=int, default=80)
    parser.add_argument("-reload", action="store_true", default=False)
    parser.add_argument("-loadfrom", action="store", type=str, default=None)
    parser.add_argument("-optimizer", action="store", type=str, default="adadelta")
    parser.add_argument("-withcxt", action="store_true", default=False)
    parser.add_argument("-hid_dim", action="store", type=int, default=256)
    parser.add_argument("-only_ul", action="store_true", default=False)
    parser.add_argument("-sl_anneal", action="store_true", default=False)
    parser.add_argument("-dt_sl", action="store", type=float, default=1.0)
    parser.add_argument("-alpha", action="store", type=float, default=1.0)
    parser.add_argument("-add_uns", action="store", type=float, default=0.2)
    parser.add_argument("-ul_num", action="store", type=float, default=10000)
    args = parser.parse_args()

    config = {}
    config["ul_num"] = args.ul_num
    config["add_uns"] = args.add_uns
    config['pure_sup'] = False
    config["epochs"] = args.epochs
    config["dt_uns"] = args.dt_uns
    config["dt_sl"] = args.dt_sl
    config["sl_anneal"] = args.sl_anneal
    config["only_sup"] = args.only_supervise
    config['start_val'] = args.start_val
    config['kl_rate'] = args.kl_rate
    config['kl_thres'] = args.kl_thres
    config['withcxt'] = args.withcxt
    config['only_ul'] = args.only_ul
    config['test'] = False
    config['index_options'] = ['word_dropout', "y_dense_p_dim", "lang", "cross_only", "only_sup", "kl_thres", "dt_uns", "add_uns"]
    config['lang'] = args.lang
    # data_sup.lang = args.lang
    # print("language: {}".format(data_sup.lang))
    config['reload'] = args.reload
    config['loadfrom'] = args.loadfrom
    config['has_ly_src'] = False
    config['cross_only'] = args.cross_only
    config["disable_kl"] = args.disable_kl
    config['alpha'] = args.alpha
    config['use_input'] = True
    config['both_gaussian'] = True
    # config['input_dim'] = 40
    config['dropout'] = 0.0
    config['word_dropout'] = args.worddrop
    # config['temperature'] = 0.1
    # config['sample_size'] = 30
    config['kl_st'] = args.kl_st
    config['activation_dense'] = 'tanh'
    config['init_dense'] = 'glorot_normal'

    config['enc_embedd_dim'] = 300  # 500 -> 300
    config['enc_hidden_dim'] = args.hid_dim
    # config['enc_contxt_dim'] = config['enc_hidden_dim']
    config['bidirectional'] = args.bidirectional  # change here
    config['shared_embed'] = args.share_emb
    # config['dec_embedd_dim']

    config['q_z_dim'] = args.z_dim  # z dim
    config['q_z_x_hidden_dim'] = None  # [enc_context_dim,128]
    config['y_dense_p_dim'] = args.y_dim  # y dim
    config['provide_init_h'] = True
    config['bias_code'] = False
    config['dec_embedd_dim'] = config['enc_embedd_dim']  # 128 -> 500
    config['dec_hidden_dim'] = config['enc_hidden_dim']
    # config['dec_contxt_dim'] = config['y_dense_p_dim'] + config['dec_hidden_dim']
    config['dec_contxt_dim'] = config['y_dense_p_dim'] + config['q_z_dim']

    config['add_sup'] = args.addsup
    config['deep_out'] = False
    # config['output_dim']
    # config['deep_out_activ']

    config['sample_stoch'] = False
    config['sample_beam'] = 8
    config['sample_argmax'] = False

    config['bigram_predict'] = False
    config['context_predict'] = True
    config['leaky_predict'] = False
    config['optimizer'] = args.optimizer

    main(config)
