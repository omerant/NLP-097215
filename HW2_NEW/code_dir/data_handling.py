from collections import defaultdict
import torch
from utils import split, get_vocabs, get_vocabs_dep_parser, WORD_IDX_IN_LINE, POS_IDX_IN_LINE, HEAD_IDX_IN_LINE, IGNORE_IDX
# from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from collections import Counter
from constants import PAD_TOKEN, SPECIAL_TOKENS, UNKNOWN_TOKEN, ROOT_TOKEN, ALPHA_DROPOUT
import os.path as osp

WORD_EMBED_SIZE = 100
POS_EMBED_SIZE = 25


class PosDataReader:
    def __init__(self, file, word_dict, pos_dict):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as f:
            for line in f:
                cur_sentence = []
                splited_words = split(line, (' ', '\n'))
                del splited_words[-1]
                for word_and_tag in splited_words:
                    cur_word, cur_tag = split(word_and_tag, '_')
                    cur_sentence.append((cur_word, cur_tag))
                self.sentences.append(cur_sentence)

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class PosDataset(Dataset):
    def __init__(self, word_dict, pos_dict, dir_path: str, subset: str,
                 padding=False, word_embeddings=None):
        super().__init__()
        self.subset = subset  # One of the following: [train, test]
        self.file = osp.join(dir_path, subset + ".wtag")
        self.datareader = PosDataReader(self.file, word_dict, pos_dict)
        self.vocab_size = len(self.datareader.word_dict)
        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
        else:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
                self.datareader.word_dict)
        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.datareader.pos_dict)

        self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        self.word_vector_dim = self.word_vectors.size(-1)
        self.sentence_lens = [len(sentence) for sentence in self.datareader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len

    @staticmethod
    def init_word_embeddings(word_dict):
        glove = Vocab(Counter(word_dict), vectors="glove.6B.300d", specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def init_pos_vocab(self, pos_dict):
        idx_pos_mappings = sorted([self.word_idx_mappings.get(token) for token in SPECIAL_TOKENS])
        pos_idx_mappings = {self.idx_word_mappings[idx]: idx for idx in idx_pos_mappings}

        for i, pos in enumerate(sorted(pos_dict.keys())):
            # pos_idx_mappings[str(pos)] = int(i)
            pos_idx_mappings[str(pos)] = int(i + len(SPECIAL_TOKENS))
            idx_pos_mappings.append(str(pos))
        print("idx_pos_mappings -", idx_pos_mappings)
        print("pos_idx_mappings -", pos_idx_mappings)
        return pos_idx_mappings, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_len_list = list()
        for sentence_idx, sentence in enumerate(self.datareader.sentences):
            words_idx_list = []
            pos_idx_list = []
            for word, pos in sentence:
                words_idx_list.append(self.word_idx_mappings.get(word))
                pos_idx_list.append(self.pos_idx_mappings.get(pos))
            sentence_len = len(words_idx_list)
            # if padding:
            #     while len(words_idx_list) < self.max_seq_len:
            #         words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
            #         pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)

        # if padding:
        #     all_sentence_word_idx = torch.tensor(sentence_word_idx_list, dtype=torch.long)
        #     all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
        #     all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
        #     return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_len_list))}


class DepDataReader:
    def __init__(self, file, word_dict, pos_dict, train_word_dict=None):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.train_word_dict = train_word_dict
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as f:
            cur_sentence=[]
            for line in f:
                if line == '\n' and len(cur_sentence) == 0:
                    break
                if line == '\n':
                    self.sentences.append(cur_sentence)
                    cur_sentence = []
                    continue
                splited_words = split(line, (' ', '\n', '\t'))
                c_word = splited_words[WORD_IDX_IN_LINE]
                c_pos = splited_words[POS_IDX_IN_LINE]
                c_head = splited_words[HEAD_IDX_IN_LINE]
                if self.train_word_dict is not None and not self.train_word_dict.get(c_word, None):
                    c_word = UNKNOWN_TOKEN
                cur_sentence.append((c_word, c_pos, c_head))

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class DepDataset(Dataset):
    def __init__(self, word_dict, pos_dict, dir_path: str, subset: str,
                 padding=False, word_embeddings=None, train_word_dict=None):
        super().__init__()
        self.word_dict = word_dict
        self.subset = subset  # One of the following: [train, test]
        self.file = osp.join(dir_path, subset + ".labeled")
        self.datareader = DepDataReader(self.file, word_dict, pos_dict, train_word_dict)
        self.vocab_size = len(self.datareader.word_dict)
        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
        else:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_idx_mapping(
                self.datareader.word_dict)
        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.datareader.pos_dict)

        self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        self.word_vector_dim = self.word_vectors.size(-1)
        self.sentence_lens = [len(sentence) for sentence in self.datareader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_dropout_prob, word_embed_idx, pos_embed_idx, head_idx, sentence_len = self.sentences_dataset[index]
        return word_dropout_prob, word_embed_idx, pos_embed_idx, head_idx, sentence_len

    # @staticmethod
    def init_word_idx_mapping(self, word_dict):
        vectors = torch.zeros((self.vocab_size+len(SPECIAL_TOKENS), WORD_EMBED_SIZE))
        word_to_idx, idx_to_word = {}, {}
        for idx, word in enumerate(SPECIAL_TOKENS+list(self.datareader.word_dict.keys())):
            # make same index mapping in train and test
            word_to_idx[word] = idx
            idx_to_word[idx] = word
        return word_to_idx, idx_to_word, vectors

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def init_pos_vocab(self, pos_dict):
        idx_pos_mappings = sorted([self.word_idx_mappings.get(token) for token in SPECIAL_TOKENS])
        pos_idx_mappings = {self.idx_word_mappings[idx]: idx for idx in idx_pos_mappings}

        for i, pos in enumerate(sorted(pos_dict.keys())):
            # pos_idx_mappings[str(pos)] = int(i)
            pos_idx_mappings[str(pos)] = int(i + len(SPECIAL_TOKENS))
            idx_pos_mappings.append(str(pos))
        print("idx_pos_mappings -", idx_pos_mappings)
        print("pos_idx_mappings -", pos_idx_mappings)
        return pos_idx_mappings, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):
        sentence_word_dropout_list = list()
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_head_idx_list = list()
        sentence_len_list = list()
        for sentence_idx, sentence in enumerate(self.datareader.sentences):
            words_dropout_list = []
            words_idx_list = []
            pos_idx_list = []
            head_idx_list = []
            for word, pos, head in sentence:
                dropout_prob = ALPHA_DROPOUT/(self.word_dict[word] + ALPHA_DROPOUT)
                # print(f'dropout_prob: {dropout_prob}')
                # print(f'count: {self.word_dict[word]}')
                words_dropout_list.append(dropout_prob)
                words_idx_list.append(self.word_idx_mappings.get(word))
                pos_idx_list.append(self.pos_idx_mappings.get(pos))
                head_idx_list.append(int(head))
            sentence_len = len(words_idx_list)
            if padding:
                while len(words_idx_list) < self.max_seq_len:
                    words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
                    pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
                    head_idx_list.append(IGNORE_IDX)

            sentence_word_dropout_list.append(torch.tensor(words_dropout_list, dtype=torch.float64, requires_grad=False))
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_head_idx_list.append(torch.tensor(head_idx_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)

        if padding:
            # all_sentence_word_idx = torch.cat(sentence_word_idx_list, dtype=torch.long)
            # all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
            # all_sentence_head_idx = torch.tensor(sentence_head_idx_list, dtype=torch.long)
            # all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
            all_sentence_word_dropout = torch.stack(sentence_word_dropout_list)
            all_sentence_word_idx = torch.stack(sentence_word_idx_list)
            all_sentence_pos_idx = torch.stack(sentence_pos_idx_list)
            all_sentence_head_idx = torch.stack(sentence_head_idx_list)
            all_sentence_len = torch.tensor(sentence_len_list, requires_grad=False)
            return TensorDataset(all_sentence_word_dropout, all_sentence_word_idx, all_sentence_pos_idx, all_sentence_head_idx, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_dropout_list,
                                                                     sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_head_idx_list,
                                                                     sentence_len_list))}


if __name__ == "__main__":
    path_train = "data_new/train.labeled"
    path_test = "data_new/test.labeled"
    paths_list_train = [path_train]
    word_dict_train, pos_dict_train = get_vocabs_dep_parser(paths_list_train)

    paths_list_all = [path_train, path_test]
    word_dict_all, pos_dict_all = get_vocabs_dep_parser(paths_list_all)

    train = DepDataset(word_dict_train, pos_dict_train, 'data_new', 'train', padding=False)
    train_dataloader = DataLoader(train, shuffle=True)
    test = DepDataset(word_dict_all, pos_dict_all, 'data_new', 'test', padding=False, train_word_dict=word_dict_train)
    test_dataloader = DataLoader(test, shuffle=False)
    print("Number of Train Tagged Sentences ", len(train))
    print("Number of Test Tagged Sentences ", len(test))


    for data in test_dataloader:
        word_dropout_prob, words_idx_tensor, pos_idx_tensor, dep_idx_tensor, sentence_length = data
        print(words_idx_tensor)
        # bern_distribution = torch.distributions.bernoulli.Bernoulli(word_dropout_prob)
        # # bern_distribution.sample()
        # words_idx_tensor[bern_distribution.sample().bool()] = train.unknown_idx
        pass

