import os
import pickle
from collections import OrderedDict, defaultdict
from re import split
from functools import partial
import features as ft
import numpy as np
from features import WordAndTagConstants
from utils import History, UNKNOWN_WORD
from scipy.sparse import csr_matrix
import gc
import multiprocessing as mp
np.random.seed(0)


class FeatureStatistics:
    def __init__(self, input_file_path, threshold, config, rare_word_num_appearence_th=3):
        #TODO: try higher rare_word_num_appearence_th
        self.file_path = input_file_path
        self.history_sentence_list = self.fill_tagged_ordered_history_list(self.file_path)
        self.num_sentences = len(self.history_sentence_list)

        # REPLACE RARE WORDS WITH UNK
        # self.rare_word_possible_tag_set = self.fill_rare_word_possible_tag_set(appearances=rare_word_num_appearence)
        self.word_possible_tag_dict = self.fill_word_possible_tag_dict()
        self.rare_word_set, self.common_word_set = self.fill_rare_word_set(appearances=rare_word_num_appearence_th)
        self.rare_words_tags = self.rare_words_possible_tags()
        # self.replace_rare_word_with_unk_in_hist()
        # call again to replace with UNK
        self.tags_set = self.fill_tags_set()

        self.word_possible_tag_set = self.fill_word_possible_tag_set()
        self.word_possible_tag_with_threshold_dict = self.fill_possible_tags_with_certainty_dict()
        # self.tag_possible_word_dict = self.fill_tag_possible_word_dict()

        self.hist_to_feature_vec_dict = dict()
        self.hist_to_all_tag_feature_matrix_dict = dict()
        self.version = 1
        self.threshold = threshold
        self.num_features = 0
        for fd_class in config.feature_dicts:
            # setattr(self, 'fd_' + str(i), ft.WordsPrefixTagsCountDict(i))
            if fd_class is ft.WordsPrefixTagsCountDict:
                for i in range(1, 5):
                    setattr(self, 'fd_' + str(fd_class) + str(i), ft.WordsPrefixTagsCountDict(i))
            elif fd_class is ft.WordsSuffixTagsCountDict:
                for i in range(1, 5):
                    setattr(self, 'fd_' + str(fd_class) + str(i), ft.WordsSuffixTagsCountDict(i))
            else:
                setattr(self, 'fd_' + str(fd_class), fd_class())
        pass

    @staticmethod
    def fill_tagged_ordered_history_list(file_path, is_test=False):
        with open(file_path) as f:
            hist_sentence_list = []
            for idx, line in enumerate(f):
                # splited_words = split(' |,\n', line[:-1]) if line[-1] == '\n' else split(' |,\n', line)  # remove \n from last part of sentence
                if not is_test:
                    splited_words = split(' |,\n', line[:-1])
                    del splited_words[-1]
                else:
                    splited_words = split(' |,\n', line[:-1]) if line[-1] == '\n' else split(' |,\n', line)
                new_sentence_hist_list = []
                for word_idx in range(len(splited_words)):
                    cword, ctag = split('_', splited_words[word_idx])

                    # check if first in sentence
                    if word_idx == 0:
                        pword = WordAndTagConstants.PWORD_SENTENCE_BEGINNING
                        ptag = WordAndTagConstants.PTAG_SENTENCE_BEGINNING
                        pptag = WordAndTagConstants.PPTAG_SENTENCE_BEGINNING
                        ppword = WordAndTagConstants.PPWORD_SENTENCE_BEGINNING
                    else:
                        prev_hist_idx = word_idx - 1
                        pword = new_sentence_hist_list[prev_hist_idx].cword
                        ptag = new_sentence_hist_list[prev_hist_idx].ctag
                        pptag = new_sentence_hist_list[prev_hist_idx].ptag
                        ppword = new_sentence_hist_list[prev_hist_idx].pword

                    # check if last in sentence
                    if word_idx + 2 < len(splited_words):
                        nword, _ = split('_', splited_words[word_idx+1])
                        nnword, _ = split('_', splited_words[word_idx+2])
                    elif word_idx + 1 < len(splited_words):
                        nword, _ = split('_', splited_words[word_idx+1])
                        nnword = WordAndTagConstants.NWORD_SENTENCE_END
                    else:
                        nword = WordAndTagConstants.NWORD_SENTENCE_END
                        nnword = WordAndTagConstants.NNWORD_SENTENCE_END
                    cur_hist = History(cword=cword, pptag=pptag, ptag=ptag, ctag=ctag, nword=nword, pword=pword,
                                       ppword=ppword, nnword=nnword)
                    new_sentence_hist_list.append(cur_hist)
                hist_sentence_list.append(new_sentence_hist_list)
        return hist_sentence_list

    @staticmethod
    def fill_comp_ordered_history_list(file_path):
        with open(file_path) as f:
            hist_sentence_list = []
            for idx, line in enumerate(f):
                splited_words = split(' |,\n', line[:-1]) if line[-1] == '\n' else split(' |,\n', line)
                new_sentence_hist_list = []
                for word in splited_words:
                    cur_hist = History(cword=word, pptag=None, ptag=None, ctag=None, nword=None, pword=None,
                                       ppword=None, nnword=None)
                    new_sentence_hist_list.append(cur_hist)
                hist_sentence_list.append(new_sentence_hist_list)
        return hist_sentence_list

    def fill_tags_set(self):
        tag_set = set()
        tag_set = tag_set | {'.', '?', '!'}
        for sentence in self.history_sentence_list:
            for hist in sentence:
                tag_set.add(hist.ctag)
        return list(tag_set)

    def fill_word_possible_tag_dict(self):
        possible_tags_dict = defaultdict(OrderedDict)
        for sentence in self.history_sentence_list:
            for hist in sentence:
                cword = hist.cword
                ctag = hist.ctag
                if possible_tags_dict[cword].get(ctag):
                    possible_tags_dict[cword][ctag] += 1
                else:
                    possible_tags_dict[cword][ctag] = 1
        return possible_tags_dict

    def fill_word_possible_tag_set(self):
        possible_tags_set = defaultdict(set)
        for sentence in self.history_sentence_list:
            for hist in sentence:
                cword = hist.cword
                ctag = hist.ctag
                possible_tags_set[cword].add(ctag)
        return possible_tags_set

    def fill_possible_tags_with_certainty_dict(self, certainty=0.995, appearances=5):
        possible_tags_with_certainty_dict = dict()
        possible_tags_with_certainty_dict['.'] = '.'
        possible_tags_with_certainty_dict['?'] = '.'
        possible_tags_with_certainty_dict['!'] = '.'
        for word, tag_count_dict in self.word_possible_tag_dict.items():
            total_count = 0
            for tag, tag_count in tag_count_dict.items():
                total_count += tag_count

            if total_count > appearances:
                for t, t_count in tag_count_dict.items():
                    if t_count / total_count > certainty:
                        possible_tags_with_certainty_dict[word] = (t, t_count)
        return possible_tags_with_certainty_dict

    def fill_tag_possible_word_dict(self):
        tag_word_dict = defaultdict(set)
        for sentence in self.history_sentence_list:
            for hist in sentence:
                cword = hist.cword
                ctag = hist.ctag
                tag_word_dict[ctag].add(cword)

        return tag_word_dict

    def fill_rare_word_possible_tag_set(self, appearances):
        rare_word_possible_tag_set = set()
        for word, tag_count_dict in self.word_possible_tag_dict.items():
            total_count = 0
            for tag, tag_count in tag_count_dict.items():
                total_count += tag_count

            if total_count < appearances:
                for t, _ in tag_count_dict.items():
                    rare_word_possible_tag_set.add(t)
        return rare_word_possible_tag_set

    def fill_rare_word_set(self, appearances):
        rare_word_set = set()
        common_word_set=set()
        for word, tag_count_dict in self.word_possible_tag_dict.items():
            total_count = 0
            for tag, tag_count in tag_count_dict.items():
                total_count += tag_count

            if total_count < appearances:
                rare_word_set.add(word)
            else:
                common_word_set.add(word)
        return rare_word_set, common_word_set

    def rare_words_possible_tags(self):
        rare_words_tags = set()
        for word in self.rare_word_set:
            rare_words_tags.update(self.word_possible_tag_dict[word].keys())
        return rare_words_tags

    def print_num_features(self):
        print('\n\n\n')
        total_feature_count = []
        feature_dicts = sorted([attr for attr in dir(self) if attr.startswith('fd')])
        for fd_name in feature_dicts:
            fd = getattr(self, fd_name)
            num_features = len(fd.dict.keys())
            total_feature_count.append(num_features)
            print(f'num of features in {fd_name}: {num_features}')

        print(f'num_total_features: {sum(total_feature_count)}')

    def _prep_args(self, num_workers, q):
        step_size = int(len(self.history_sentence_list) / num_workers)
        # each argument tuple is (hist_sentence_list, q, self)
        args_list = [(self.history_sentence_list[i * step_size: (i+1)*step_size], q, self) for i in range(num_workers)]
        return args_list

    @staticmethod
    def _calc_hist_dicts(hist_sentence_list: [[History]], q: mp.Queue, self):
        """ method that calculates  two dictionaries - hist_to_all_tag_feature_matrix_dict and hist_to_feature_vec_dict

        :param hist_sentence_list: sentenc
        :param q:
        :param self:
        :return:
        """
        hist_to_feature_vec_dict = dict()
        hist_to_all_tag_feature_matrix_dict = dict()
        for idx_sentence, sentence in enumerate(hist_sentence_list):
            if idx_sentence % 500 == 0:
                gc.collect()
                print(f'filling sentence number {idx_sentence}')
            for hist in sentence:
                tag_set = self.tags_set
                cur_feature_vecs = []
                hist_to_feature_vec_dict[hist] = csr_matrix(self.get_non_zero_feature_vec_indices_from_history(hist))
                for ctag in tag_set:
                    new_hist = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag,
                                       ctag=ctag, nword=hist.nword, pword=hist.pword,
                                       nnword=hist.nnword, ppword=hist.ppword)

                    cur_feature_vecs.append(self.get_non_zero_feature_vec_indices_from_history(new_hist))

                key_all_tag_hist = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag,
                                           ctag=None, nword=hist.nword, pword=hist.pword,
                                           nnword=hist.nnword, ppword=hist.ppword)

                # fill dict that contains matrices with dim num_tagsXnum_features, it will be used to speed up operations
                if hist_to_all_tag_feature_matrix_dict.get(key_all_tag_hist, None) is None:
                    sparse_res = csr_matrix(cur_feature_vecs)

                    # sparse_mem = sparse_res.data.nbytes + sparse_res.indptr.nbytes + sparse_res.indices.nbytes
                    hist_to_all_tag_feature_matrix_dict[key_all_tag_hist] = sparse_res

        q.put((hist_to_all_tag_feature_matrix_dict, hist_to_feature_vec_dict))

    def fill_all_possible_tags_dict(self, hist_ft_dict_path, hist_dict_name, num_workers=4):
        """

        :param hist_ft_dict_path:
        :param hist_dict_name:
        :param num_workers:
        :return:
        """
        print('filling all_possible_prev_tags_dict')
        dict_folder = 'hist_feature_dict'
        if not os.path.isdir(dict_folder):
            os.mkdir(dict_folder)
        # each hist subset is independent - calculate with num_workers
        p = mp.Pool(num_workers)
        manager = mp.Manager()
        q = manager.Queue()
        args = self._prep_args(num_workers=num_workers, q=q)
        p.starmap(func=self._calc_hist_dicts, iterable=args)
        p.close()
        p.join()
        # now q contains tuples of (hist_to_feature_vec_dict, hist_to_all_tag_feature_matrix_dict)
        res_list = []
        while q.qsize() > 0:
            res_list.append(q.get())

        for hist_all_tag_feature_mat_dict, hist_feature_vec_dict in res_list:
            self.hist_to_feature_vec_dict.update(hist_feature_vec_dict)
            self.hist_to_all_tag_feature_matrix_dict.update(hist_all_tag_feature_mat_dict)

        if not os.path.isdir(hist_ft_dict_path):
            os.makedirs(hist_ft_dict_path)
        full_path = os.path.join(hist_ft_dict_path, hist_dict_name)
        with open(full_path, "wb") as f:
            p = pickle.Pickler(f)
            p.fast = True
            p.dump((self.hist_to_feature_vec_dict, self.hist_to_all_tag_feature_matrix_dict))

        print(f'total keys in all possible tags dict: {len(self.hist_to_feature_vec_dict.keys())}')

    def load_all_possible_tags_dict(self, path):
        print('loading all_possible_prev_tags_dict')
        with open(path, 'rb') as f:
            self.hist_to_feature_vec_dict, self.hist_to_all_tag_feature_matrix_dict = pickle.load(f)
        print('finished loading all_possible_prev_tags_dict')

    def fill_num_features(self):
        total_feature_count = 0
        feature_dicts = sorted([attr for attr in dir(self) if attr.startswith('fd')])
        for fd_name in feature_dicts:
            fd = getattr(self, fd_name)
            num_features = len(fd.dict.keys())
            total_feature_count += num_features
        self.num_features = total_feature_count

    def fill_feature_dicts(self):
        feature_dicts = sorted([attr for attr in dir(self) if attr.startswith('fd')])
        for fd_name in feature_dicts:
            fd = getattr(self, fd_name)
            fd.fill_dict(self.history_sentence_list)

    @staticmethod
    def _filter_dict(d, threshold):
        return OrderedDict({k: v for k, v in sorted(d.items(), key=lambda x: x[0]) if d[k] > threshold})

    def filter_features_by_threshold(self):
        filter_dict = partial(self._filter_dict, threshold=self.threshold)
        feature_dicts = sorted([attr for attr in dir(self) if attr.startswith('fd')])
        for fd_name in feature_dicts:
            fd = getattr(self, fd_name)
            if fd_name != 'fd_word_tag':
                fd.dict = filter_dict(fd.dict)

    def get_non_zero_feature_vec_indices_from_history(self, history: ft.History):
        sparse_feature_vec = OrderedDict()
        feature_dicts = sorted([attr for attr in dir(self) if attr.startswith('fd')])
        start_idx = 0
        for fd_name in feature_dicts:
            fd = getattr(self, fd_name)
            idx_in_dict, val = fd.get_feature_index_and_count_from_history(history)
            # there is a feature in dict for this history
            if idx_in_dict != -1:
                cur_idx = start_idx + idx_in_dict
                sparse_feature_vec[cur_idx] = 1
            # there is no feature in dict for this history
            start_idx += len(fd.dict.keys())
        feature_vec = np.full(self.num_features, 0, np.uintc)

        for k, v in sparse_feature_vec.items():
            feature_vec[k] = v

        return feature_vec

    def pre_process(self, fill_possible_tag_dict: bool = True):
        self.fill_feature_dicts()
        self.filter_features_by_threshold()
        self.fill_num_features()
        self.print_num_features()

        hist_ft_dict_path = os.path.join('hist_feature_dict', f'th={self.threshold}')
        hist_dict_name = f'version={self.version}_threshold={self.threshold}'

        if fill_possible_tag_dict:
            self.fill_all_possible_tags_dict(hist_ft_dict_path, hist_dict_name)
        else:
            dump_path = os.path.join(hist_ft_dict_path, hist_dict_name)
            self.load_all_possible_tags_dict(path=dump_path)
