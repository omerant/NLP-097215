import os
import pickle
from collections import OrderedDict, defaultdict
from re import split
from functools import partial
import features as ft
import numpy as np
from features import History, WordAndTagConstants


class FeatureStatistics:
    def __init__(self, input_file_path, threshold=3):
        self.file_path = input_file_path
        self.history_sentence_list = self.fill_ordered_history_list(self.file_path)
        self.num_sentences = len(self.history_sentence_list)
        self.tags_set = self.fill_tags_set()
        self.word_possible_tag_dict = self.fill_word_possible_tag_dict()
        self.word_possible_tag_set = self.fill_word_possible_tag_set()
        self.word_possible_tag_with_threshold_dict = self.fill_possible_tags_with_certainty_dict()
        self.tag_possible_word_dict = self.fill_tag_possible_word_dict()
        self.all_possible_tags_dict = dict()
        self.version = 1
        self.threshold = threshold
        self.num_features = 0

        # Init all features dictionaries - each feature dict's name should start with fd
        # fill dict for 1-3 ngrams
        self.fd_trigram_tags = ft.TrigramTagsCountDict()
        self.fd_bigram_tags = ft.BigramTagsCountDict()
        self.fd_unigram_tags = ft.UnigramTagsCountDict()
        self.fd_word_tag = ft.WordsTagsCountDict()
        # fill dict for each prefix len
        for i in range(1, 7):
            setattr(self, 'fd_words_prefix'+str(i), ft.WordsPrefixTagsCountDict(i))

        # fill dict for each suffix len
        for i in range(1, 7):
            setattr(self, 'fd_words_suffix'+str(i), ft.WordsSuffixTagsCountDict(i))

        self.fd_pword_ctag = ft.PrevWordCurrTagCountDict()
        self.fd_nword_ctag = ft.NextWordCurrTagCountDict()
        self.fd_ctag_pptag = ft.SkipBigramCountDict()
        # letters digits
        self.fd_has_first_capital_letter = ft.HasFirstCapitalLetterDict()
        self.fd_has_all_capital_letters = ft.HasAllCapitalLettersDict()
        self.fd_has_digit = ft.HasDigitDict()
        self.fd_has_only_digits = ft.HasOnlyDigitDict()
        self.fd_contains_letter = ft.ContainsLetterDict()
        self.fd_has_only_letters = ft.ContainsOnlyLettersDict()
        self.fd_contains_hyphen = ft.ContainsHyphenDict()
        # first last
        self.fd_is_first_word = ft.IsFirstWordDict()
        self.fd_is_lat_word = ft.IsLastWordDict()
        # symbols
        self.fd_has_symbol = ft.ContainsSymbolDict()
        self.fd_all_symbol = ft.ContainsOnlySymbolsDict()

        #word length
        for i in range(1, 14):
            setattr(self, 'fd_word_length'+str(i), ft.WordsLengthDict(i))

    @staticmethod
    def fill_ordered_history_list(file_path):
        with open(file_path) as f:
            hist_sentence_list = []
            for idx, line in enumerate(f):
                splited_words = split(' |,\n', line[:-1]) if line[-1] == '\n' else split(' |,\n', line)  # remove \n from last part of sentence
                new_sentence_hist_list = []
                for word_idx in range(len(splited_words)):
                    cword, ctag = split('_', splited_words[word_idx])

                    # check if first in sentence
                    if word_idx == 0:
                        pword = WordAndTagConstants.PWORD_SENTENCE_BEGINNING
                        ptag = WordAndTagConstants.PTAG_SENTENCE_BEGINNING
                        pptag = WordAndTagConstants.PPTAG_SENTENCE_BEGINNING

                    else:
                        prev_hist_idx = word_idx - 1
                        pword = new_sentence_hist_list[prev_hist_idx].cword
                        ptag = new_sentence_hist_list[prev_hist_idx].ctag
                        pptag = new_sentence_hist_list[prev_hist_idx].ptag

                    # check if last in sentence
                    if word_idx + 1 < len(splited_words):
                        nword, _ = split('_', splited_words[word_idx+1])
                    else:
                        nword = WordAndTagConstants.NWORD_SENTENCE_END
                    cur_hist = History(cword=cword, pptag=pptag, ptag=ptag, ctag=ctag, nword=nword, pword=pword)
                    new_sentence_hist_list.append(cur_hist)
                hist_sentence_list.append(new_sentence_hist_list)
        return hist_sentence_list

    def fill_tags_set(self):
        tag_set = set()
        for sentence in self.history_sentence_list:
            for hist in sentence:
                tag_set.add(hist.ctag)
        return tag_set

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

    def fill_all_possible_tags_dict(self, hist_ft_dict_path, hist_dict_name):
        print('filling all_possible_prev_tags_dict')
        dict_folder = 'hist_feature_dict'
        if not os.path.isdir(dict_folder):
            os.mkdir(dict_folder)

        for idx, sentence in enumerate(self.history_sentence_list):
            for hist in sentence:
                tag_set = self.word_possible_tag_set[hist.cword]
                for ctag in tag_set:
                    new_hist = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag,
                                       ctag=ctag, nword=hist.nword, pword=hist.pword)
                    if not self.all_possible_tags_dict.get(new_hist, None):
                        self.all_possible_tags_dict[new_hist] = self.get_non_zero_sparse_feature_vec_indices_from_history(new_hist)
        if not os.path.isdir(hist_ft_dict_path):
            os.makedirs(hist_ft_dict_path)
        full_path = os.path.join(hist_ft_dict_path, hist_dict_name)
        with open(full_path, "wb") as f:
            p = pickle.Pickler(f)
            p.fast = True
            p.dump(self.all_possible_tags_dict)

        print(f'total keys in all possible tags dict: {len(self.all_possible_tags_dict.keys())}')

    def load_all_possible_tags_dict(self, path):
        print('loading all_possible_prev_tags_dict')
        with open(path, 'rb') as f:
            self.all_possible_tags_dict = pickle.load(f)
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
            fd.dict = filter_dict(fd.dict)

    def get_non_zero_sparse_feature_vec_indices_from_history(self, history: ft.History):
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

        non_zero_indices = np.nonzero(feature_vec)
        return non_zero_indices

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
