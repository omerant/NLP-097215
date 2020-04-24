import os
import pickle
from collections import OrderedDict
from re import split
from functools import partial
import features
import numpy as np
from features import History
import gc


class FeatureStatistics:
    def __init__(self, file_path, threshold=3):
        self.n_total_features = 0  # Total number of features accumulated
        self.file_path = file_path
        self.tag_ordered_list = []
        self.word_ordered_list = []
        self.history_ordered_list = []
        self.history_sentence_list = []
        self.batch_hist_list = None
        self.num_features = None
        self.num_sentences = None
        self.tags_set = set()
        self.all_possible_tags_dict = dict()
        self.version = 2
        self.threshold = threshold

        # Init all features dictionaries - each feature dict's name should start with fd
        self.fd_words_tags_count_dict = features.WordsTagsCountDict()
        # fill dict for each prefix len
        self.fd_words_prefix1_tags_count_dict = features.WordsPrefixTagsCountDict(pref_len=1)
        self.fd_words_prefix2_tags_count_dict = features.WordsPrefixTagsCountDict(pref_len=2)
        self.fd_words_prefix3_tags_count_dict = features.WordsPrefixTagsCountDict(pref_len=3)
        self.fd_words_prefix4_tags_count_dict = features.WordsPrefixTagsCountDict(pref_len=4)
        # fill dict for each suffix len
        self.fd_words_suffix1_tags_count_dict = features.WordsSuffixTagsCountDict(suff_len=1)
        self.fd_words_suffix2_tags_count_dict = features.WordsSuffixTagsCountDict(suff_len=2)
        self.fd_words_suffix3_tags_count_dict = features.WordsSuffixTagsCountDict(suff_len=3)
        self.fd_words_suffix4_tags_count_dict = features.WordsSuffixTagsCountDict(suff_len=4)
        # fill dict for 1-3 ngrams
        self.fd_trigram_tags_count_dict = features.TrigramTagsCountDict()
        self.fd_bigram_tags_count_dict = features.BigramTagsCountDict()
        self.fd_unigram_tags_count_dict = features.UnigramTagsCountDict()

        self.fd_prev_word_curr_tag_count_dict = features.PrevWordCurrTagCountDict()
        self.fd_next_word_curr_tag_count_dict = features.NextWordCurrTagCountDict()
        # letters and digits
        self.fd_has_first_capital_letter_dict = features.HasFirstCapitalLetterDict()
        self.fd_has_all_capital_letters_dict = features.HasAllCapitalLettersDict()
        self.fd_has_digit_dict = features.HasDigitDict()
        self.fd_contains_letter = features.ContainsLetterDict()
        # has same tag more then x% of the times
        self.fd_has_same_tag_most_times_dict = features.HasSameTagDict()
        # has same tag always
        self.fd_has_same_tag_always_dict = features.HasSameTagAlwaysDict()
        # contains hyphen
        self.fd_contains_hyphen_dict = features.ContainsHyphenDict()
        # is first, is last
        self.fd_is_first_word = features.IsFirstWordDict()
        self.fd_is_last_word = features.IsLastWordDict()

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

    def fill_word_and_tag_ordered_lists(self):
        with open(self.file_path) as f:
            for line in f:
                splited_words = split(' |,\n', line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = split('_', splited_words[word_idx])

                    self.word_ordered_list.append(cur_word)
                    self.tag_ordered_list.append(cur_tag)

    def fill_words_with_one_tag_only(self):
        res_dict = OrderedDict()
        count_dict = OrderedDict()
        words_with_many_tags = set()
        for word, tag in zip(self.word_ordered_list, self.tag_ordered_list):
            if word in words_with_many_tags:
                continue

            res_tag = res_dict.get(word, None)
            if res_tag:
                if tag != res_tag:
                    words_with_many_tags.add(word)
                    del res_dict[word]
                    del count_dict[word]
                else:
                    count_dict[word] += 1
            else:
                res_dict[word] = tag
                count_dict[word] = 1

        # count_dict = {k: count_dict[k] for k in sorted(count_dict.values())}
        count_list = sorted([(k, v, res_dict[k]) for k, v in count_dict.items()], key=lambda x: x[1], reverse=True)
        num_over_100 = sum([item[1] for item in count_list if item[1] >= 100])
        pass

    def fill_tags_set(self):
        self.tags_set = set(self.tag_ordered_list) | {'*'}

    def fill_all_possible_tags_dict(self):
        print('filling all_possible_prev_tags_dict')
        dict_folder = 'hist_feature_dict'
        if not os.path.isdir(dict_folder):
            os.mkdir(dict_folder)
        for idx, hist in enumerate(self.history_ordered_list):
            if idx % 100 == 0:
                print(f'hist {idx} from{len(self.history_ordered_list)}')

            for ctag in self.tags_set:
                new_hist = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag,
                                   ctag=ctag, nword=hist.nword, pword=hist.pword)
                self.all_possible_tags_dict[new_hist] = self.get_non_zero_sparse_feature_vec_indices_from_history(new_hist)
        dump_path = os.path.join(dict_folder, f'version={self.version}_threshold={self.threshold}')
        gc.collect()
        with open(dump_path, "wb") as f:
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
        total_feature_count = []
        feature_dicts = sorted([attr for attr in dir(self) if attr.startswith('fd')])
        for fd_name in feature_dicts:
            fd = getattr(self, fd_name)
            num_features = len(fd.dict.keys())
            total_feature_count.append(num_features)
        self.num_features = sum(total_feature_count)

    def fill_ordered_history_list(self):
        with open(self.file_path) as f:
                for line in f:
                    cword = '*'
                    pptag = '*'
                    ptag = '*'
                    ctag = '*'
                    nword = None
                    pword = '*'
                    ppword = '*'
                    splited_words = split(' |,\n', line)
                    del splited_words[-1]
                    new_sentence = []
                    for word_idx in range(len(splited_words)):
                        cur_word, cur_tag = split('_', splited_words[word_idx])

                        ppword = pword
                        pword = cword
                        pptag = ptag
                        ptag = ctag
                        cword = cur_word
                        ctag = cur_tag
                        if word_idx + 1 < len(splited_words):
                            nword, _ = split('_', splited_words[word_idx+1])
                        else:
                            nword = '*'
                        cur_hist = History(cword=cword, pptag=pptag, ptag=ptag, ctag=ctag, nword=nword, pword=pword)
                        new_sentence.append(cur_hist)
                        self.history_ordered_list.append(cur_hist)
                    self.history_sentence_list.append(new_sentence)

    def fill_feature_dicts(self):
        feature_dicts = sorted([attr for attr in dir(self) if attr.startswith('fd')])
        for fd_name in feature_dicts:
            fd = getattr(self, fd_name)
            fd.fill_dict(self.word_ordered_list, self.tag_ordered_list, self.file_path)

    @staticmethod
    def _filter_dict(d, threshold):
        return OrderedDict({k: v for k, v in sorted(d.items(), key=lambda x: x[0]) if d[k] > threshold})

    def filter_features_by_threshold(self):
        filter_dict = partial(self._filter_dict, threshold=self.threshold)
        feature_dicts = sorted([attr for attr in dir(self) if attr.startswith('fd')])
        for fd_name in feature_dicts:
            fd = getattr(self, fd_name)
            fd.dict = filter_dict(fd.dict)

    def get_non_zero_sparse_feature_vec_indices_from_history(self, history: features.History):
        sparse_feature_vec = OrderedDict()
        feature_dicts = sorted([attr for attr in dir(self) if attr.startswith('fd')])
        start_idx = 0
        for fd_name in feature_dicts:
            fd = getattr(self, fd_name)
            idx_in_dict, val = fd.get_feature_index_from_history(history)
            # there is a feature in dict for this history
            if idx_in_dict != -1:
                cur_idx = start_idx + idx_in_dict
                sparse_feature_vec[cur_idx] = 1  # TODO: use val instead of just boolean
            # there is no feature in dict for this history
            else:
                pass
            # print(start_idx)
            start_idx += len(fd.dict.keys())
        feature_vec = np.full(self.num_features, 0, np.uintc)

        for k, v in sparse_feature_vec.items():
            feature_vec[k] = v

        non_zero_indices = np.nonzero(feature_vec)
        return non_zero_indices

    def pre_process(self):
        self.fill_word_and_tag_ordered_lists()
        # self.get_words_with_unique_tag()
        # filter invalid words and tags

        self.fill_words_with_one_tag_only()

        self.fill_tags_set()
        # fill feature dicts
        self.fill_feature_dicts()

        # filter features by threshold
        self.filter_features_by_threshold()
        self.fill_num_features()

        self.fill_ordered_history_list()
        self.fill_num_sentences()
        # self.get_batch_history_list()
        self.print_num_features()

        # self.fill_all_possible_tags_dict()
        dump_path = os.path.join('hist_feature_dict', f'version={self.version}_threshold={self.threshold}')
        self.load_all_possible_tags_dict(path=dump_path)

    def create_history_list(self):
        self.fill_ordered_history_list()

    def get_batch_history_list(self, num_sentences_in_batch=1):
        # total of 5000 sentences
        hist_batch_list = [[] for i in range(round(5000/num_sentences_in_batch) + 1)]
        curr_batch_idx = 0
        cur_num_sentence_in_batch = 0
        for i in range(len(self.history_ordered_list)):
            hist_batch_list[curr_batch_idx].append(self.history_ordered_list[i])

            if self.history_ordered_list[i].nword == '*':
                cur_num_sentence_in_batch += 1

            if cur_num_sentence_in_batch == num_sentences_in_batch:
                curr_batch_idx += 1
                cur_num_sentence_in_batch = 0

        self.batch_hist_list = hist_batch_list

    def fill_num_sentences(self):
        num_sentences = 0
        for i in range(len(self.history_ordered_list)):
            if self.history_ordered_list[i].pword == '*':
                num_sentences += 1
        self.num_sentences = num_sentences

    def get_feature_dict_names(self):
        feature_names = sorted([attr for attr in dir(self) if attr.startswith('fd')])
        feature_names_str = ''
        for feature_name in feature_names:
            feature_names_str += feature_name + '\n'
        return feature_names_str


if __name__ == '__main__':
    train1_path = 'data/train1.wtag'
    feature_statistics = FeatureStatistics(train1_path)
    feature_statistics.pre_process()

