import torch
from model import DnnSepParser
from utils import get_vocabs_dep_parser, constract_line
from data_handling import DepDataset, DepDataReader
from torch.utils.data.dataloader import DataLoader
from constants import UNKNOWN_TOKEN


def calculate_comp(model, comp_dataloader, idx_word_mappings, idx_pos_mappings, output_path):

    dep_data_reader = DepDataReader(file='data_new/comp.unlabeled', word_dict={}, pos_dict={})
    with torch.no_grad():
        with open(output_path, 'w') as out_file:

            for sentence, input_data in zip(dep_data_reader.sentences, comp_dataloader):
                cur_idx_in_sentence = 1
                _, words_idx_tensor, pos_idx_tensor, _, _ = input_data
                words = [idx_word_mappings[word_idx.item()] for word_idx in words_idx_tensor.squeeze(0)][1:]
                tags = [idx_pos_mappings[tag_idx.item()] for tag_idx in pos_idx_tensor.squeeze(0)][1:]
                _, our_heads, _,  = model(words_idx_tensor, pos_idx_tensor, calc_mst=True)

                sentence = sentence[1:]
                for word, tag, head in zip(words, tags, our_heads[1:]):
                    if word == UNKNOWN_TOKEN:
                        word = sentence[cur_idx_in_sentence-1][0]
                    out_file.write(constract_line(cur_idx_in_sentence, word, tag, head))
                    cur_idx_in_sentence += 1
                    out_file.write('\n')
                out_file.write('\n')


def gen_comp_files():
    path_train = "data_new/train.labeled"
    path_test = "data_new/test.labeled"

    paths_list_all = [path_train, path_test]
    word_dict_all, pos_dict_all = get_vocabs_dep_parser(paths_list_all)

    train = DepDataset(word_dict_all, pos_dict_all, 'data_new', 'train.labeled', padding=False)

    test = DepDataset(word_dict_all, pos_dict_all, 'data_new', 'comp.unlabeled', padding=False, train_word_dict=word_dict_all,
                      is_comp=True)

    test_dataloader = DataLoader(test, shuffle=False)
    word_vocab_size = len(train.word_idx_mappings)
    tag_vocab_size = len(train.pos_idx_mappings)
    # init model
    _gen_comp_files_model1(word_vocab_size, tag_vocab_size, test_dataloader, test)
    _gen_comp_files_model2(word_vocab_size, tag_vocab_size, test_dataloader, test)


def _gen_comp_files_model1(word_vocab_size, tag_vocab_size, test_dataloader, test_dataset):
    model = DnnSepParser(word_emb_dim=100, tag_emb_dim=25, num_layers=2, word_vocab_size=word_vocab_size,
                         tag_vocab_size=tag_vocab_size, hidden_fc_dim=100)
    # load weights
    path = 'checkpoints/best_first_model/DnnDepParser_word_emb-100_tag_emb-25_num_stack2_50.pth'
    train_info = torch.load(path, map_location=torch.device('cpu'))
    word_emb = train_info['word_embedding']
    tag_emb = train_info['tag_embedding']
    model.load_state_dict(train_info['net'])
    model.load_embedding(word_emb, tag_emb)
    calculate_comp(model=model, comp_dataloader=test_dataloader, idx_word_mappings=test_dataset.idx_word_mappings,
                   idx_pos_mappings=test_dataset.idx_pos_mappings, output_path='comp_m1_311773915.wtag')


def _gen_comp_files_model2(word_vocab_size, tag_vocab_size, test_dataloader, test_dataset):
    model = DnnSepParser(word_emb_dim=250, tag_emb_dim=25, num_layers=2, word_vocab_size=word_vocab_size,
                         tag_vocab_size=tag_vocab_size, hidden_fc_dim=500)
    # load weights
    path = 'checkpoints/best_second_model/DnnDepParser_word_emb-250_tag_emb-25_num_stack2_hidden_fc_dim-500_50.pth'
    train_info = torch.load(path, map_location=torch.device('cpu'))
    word_emb = train_info['word_embedding']
    tag_emb = train_info['tag_embedding']
    model.load_state_dict(train_info['net'])
    model.load_embedding(word_emb, tag_emb)
    calculate_comp(model=model, comp_dataloader=test_dataloader, idx_word_mappings=test_dataset.idx_word_mappings,
                   idx_pos_mappings=test_dataset.idx_pos_mappings, output_path='comp_m2_311773915.wtag')


if __name__ == '__main__':
    gen_comp_files()