import helper

from torch.utils.data import DataLoader

from dataset import *

morph_data = MorphologyDatasetTask3(csv_file='../data/files/turkish-task3-dev.csv', language='turkish',
                                    root_dir='../data/files')

idx_2_char = helper.load_file('../data/pickles/idx_2_char')
char_2_idx = helper.load_file('../data/pickles/char_2_idx')
idx_2_desc = helper.load_file('../data/pickles/idx_2_desc')
desc_2_idx = helper.load_file('../data/pickles/desc_2_idx')
msd_types  = helper.load_file('../data/pickles/msd_options')  # label types

morph_data.set_vocabulary(char_2_idx, idx_2_char, desc_2_idx, idx_2_desc, msd_types)



# print(len(morph_data))

train_dataloader = DataLoader(morph_data, batch_size=5, shuffle=True, num_workers=2)

for i_batch, sample_batched in enumerate(train_dataloader):
    print("-----")
    print(sample_batched['msd'])
    print(sample_batched['msd'].size())
    print("------")

    t = torch.transpose(sample_batched['msd'], 0, 1)

    print(t)
    print(t.size())

    print('----')
    print(morph_data.get_unprocessed_strings(i_batch))

    break




# print(morph_data.max_seq_len)
