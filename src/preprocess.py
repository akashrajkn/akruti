import os

from data import *

if __name__ == '__main__':

    # task 3 files
    path = '../data/files/task3_test'
    out  = read_file(path, task=3)

    print(out)

    save_file('../data/pickles/task3_test.pkl', out)
