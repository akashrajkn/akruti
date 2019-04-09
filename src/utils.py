import os
import pickle


def save_file(path, dat):
    '''
    Dump pickle file
    Args:
        path: file path to save
        dat : python struct to be saved
    '''

    with open(path, 'wb') as of:
        pickle.dump(dat, of, protocol=pickle.HIGHEST_PROTOCOL)


def max_sequence_length(filepath):
    '''
    Return the length of the longest source/target sequence
    '''

    max_len = 0
    all_out = read_task_3(filepath)

    for triplet in all_out:
        source_len = len(triplet['source_form'])
        target_len = len(triplet['target_form'])

        if (source_len > max_len):
            max_len = source_len

        if (target_len > max_len):
            max_len = target_len

    return max_len

def load_file(path):
    '''
    load pickle File
    Args:
        path: file path
    Returns:
        out : pickle loaded
    '''

    with open(path, 'rb') as f:
        out = pickle.load(f)

    return out


def read_file(path, task):
    '''
    Read file and return respective dictionary
    Args:
        path: file path
        task: task number

    Returns:
        out : dictionary
    '''
    if path is None:
        print('File path not specified')
        return

    if   task == 1:
        out = read_task_1(path)
    elif task == 2:
        out = read_task_2(path)
    elif task == 3:
        out = read_task_3(path)
    else:
        print('Task number is wrong')

    return out


def process_MSD(msd):
    '''
    Process morphosyntactic descriptions (MSDs) in the input sentence
    Args:
        msd: string containing different MSDs
    Returns:
        out: dict with different msds
    '''

    out = {}
    msds = msd.strip().split(',')

    for m in msds:
        current = m.strip().split('=')
        out[current[0]] = current[1]

    return out


def read_task_1(path):
    '''
    Task 1 data: lemma  MSD  target_form
    Args:
        path: file path
    Returns:
        out : dictionary
    '''
    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()

    out       = []
    sentences = source.strip().split('\n')

    for sentence in sentences:
        line  = sentence.strip().split('\t')

        if len(line) > 3:
            print('Something wrong with line: {}'.format(sentence))
            continue

        out.append({
            'lemma'      : line[0],
            'MSD'        : process_MSD(line[1]),
            'target_form': line[2]
        })

    return out


def read_task_3(path):
    '''
    Task 3 data: source_form  MSD  target_form
    Args:
        path: file path
    Returns:
        out : dictionary
    '''
    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()

    out       = []
    sentences = source.strip().split('\n')

    for sentence in sentences:
        line  = sentence.strip().split('\t')

        if len(line) > 3:
            print('Something wrong with line: {}'.format(sentence))
            continue

        out.append({
            'source_form': line[0],
            'MSD'        : process_MSD(line[1]),
            'target_form': line[2]
        })

    return out
