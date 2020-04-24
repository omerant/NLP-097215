from collections import OrderedDict, namedtuple
from abc import abstractmethod, ABC
from common_features import common_tags
import re
from constant_tag_word_sets import WordConstantTagSets

History = namedtuple('History', 'cword, pptag, ptag, ctag, nword, pword')


class FeatureDict(ABC):
    def __init__(self):
        self.dict = OrderedDict()

    @abstractmethod
    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        pass

    @abstractmethod
    def get_feature_index_from_history(self, history: History):
        pass

    def insert_key(self, key):
        if key not in self.dict.keys():
            self.dict[key] = 1
        else:
            self.dict[key] += 1

    def get_key_index(self, key):
        if key not in self.dict.keys():
            return -1, -1

        for idx, k in enumerate(self.dict.keys()):
            if key == k:
                return idx, self.dict[key]


class WordsTagsCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Extract out of text all word/tag pairs <w_i, t_i>
            fill all word/tag pairs with index of appearance
        """
        for cur_word, cur_tag in zip(word_ordered_list, tag_ordered_list):
            self.insert_key((cur_word, cur_tag))

    def get_feature_index_from_history(self, history: History):
        key = (history.cword, history.ctag)
        return self.get_key_index(key)


class WordsPrefixTagsCountDict(FeatureDict):
    def __init__(self, pref_len):
        super().__init__()
        self.pref_len = pref_len

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Extract out of text all word_prefix/tag pairs for word_prefix| <=4 - <pref_w_i, t_i>
            fill all word_prefix/tag pairs with index of appearance

        """
        for cur_word, cur_tag in zip(word_ordered_list, tag_ordered_list):
            if len(cur_word) < self.pref_len:
                continue

            pref = cur_word[:self.pref_len]
            self.insert_key((pref, cur_tag))

    def get_feature_index_from_history(self, history: History):
        cur_word = history.cword
        cur_tag = history.ctag
        if len(cur_word) < self.pref_len:
            return -1, -1
        cur_pref = cur_word[:self.pref_len]
        key = (cur_pref, cur_tag)
        return self.get_key_index(key)


class WordsSuffixTagsCountDict(FeatureDict):
    def __init__(self, suff_len=1):
        super().__init__()
        self.suff_len = suff_len

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Extract out of text all word_suffix/tag pairs for |word_suffix| <=4 - <suff_w_i, t_i>
            fill all word_suffix/tag pairs with index of appearance
        """
        for cur_word, cur_tag in zip(word_ordered_list, tag_ordered_list):
            if len(cur_word) < self.suff_len:
                continue

            pref = cur_word[-self.suff_len:]
            self.insert_key((pref, cur_tag))

    def get_feature_index_from_history(self, history: History):
        cur_word = history.cword
        cur_tag = history.ctag
        if len(cur_word) < self.suff_len:
            return -1, -1
        cur_suff = cur_word[-self.suff_len:]
        key = (cur_suff, cur_tag)
        return self.get_key_index(key)


class TrigramTagsCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Extract out of text ordered tag triplets - <t_i-2, t_i-1, t_i>
            fill all ordered tag triplets with index of appearance
        """
        for i in range(len(tag_ordered_list)):
            cur_triplet = None
            if i == 0:
                cur_triplet = ('*', '*', tag_ordered_list[i])
            elif i == 1:
                cur_triplet = ('*', tag_ordered_list[i-1], tag_ordered_list[i])
            else:
                cur_triplet = (tag_ordered_list[i-2], tag_ordered_list[i-1], tag_ordered_list[i])

            self.insert_key(cur_triplet)

    def get_feature_index_from_history(self, history: History):
        key = (history.pptag, history.ptag, history.ctag)
        return self.get_key_index(key)


class BigramTagsCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Extract out of text ordered tag pairs - <t_i-1, t_i>
            fill all ordered tag pairs with index of appearance
        """
        for i in range(len(tag_ordered_list)):
            cur_pair = None
            if i == 0:
                cur_pair = ('*', tag_ordered_list[i])
            else:
                cur_pair = (tag_ordered_list[i - 1], tag_ordered_list[i])

            self.insert_key(cur_pair)

    def get_feature_index_from_history(self, history: History):
        key = (history.ptag, history.ctag)
        return self.get_key_index(key)


class UnigramTagsCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Extract out of text tags - <t_i>
            fill all tags with index of appearance
        """
        for i in range(len(tag_ordered_list)):
            cur_tag = tag_ordered_list[i]
            self.insert_key(cur_tag)

    def get_feature_index_from_history(self, history: History):
        key = history.ctag
        return self.get_key_index(key)


class PrevWordCurrTagCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Extract out of text all word/tag pairs - <w_i-1, t_i>
            fill all word/tag pairs with index of appearance
        """
        for i in range(len(tag_ordered_list)):
            cur_tag = tag_ordered_list[i]
            prev_word = None
            if i == 0:
                prev_word = '*'
            else:
                prev_word = word_ordered_list[i-1]

            res = (prev_word, cur_tag)
            self.insert_key(res)

    def get_feature_index_from_history(self, history: History):
        key = (history.pword, history.ctag)
        return self.get_key_index(key)


class NextWordCurrTagCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Extract out of text all word/tag pairs - <w_i+1, t_i>
            fill all word/tag pairs with index of appearance
        """
        for i in range(len(tag_ordered_list)):
            cur_tag = tag_ordered_list[i]
            next_word = None
            if i == len(tag_ordered_list) - 1:
                next_word = '*'
            else:
                next_word = word_ordered_list[i+1]

            res = (next_word, cur_tag)
            self.insert_key(res)

    def get_feature_index_from_history(self, history: History):
        key = (history.nword, history.ctag)
        return self.get_key_index(key)


class HasFirstCapitalLetterDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'begins_with_capital_letter'

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Extract out of text all words that contain capital letters - <w_i | w_i contains capital letter>
            fill all words with capital letters with index of appearance
        """
        for i in range(len(word_ordered_list)):
            cur_word = word_ordered_list[i]
            if bool(re.search(r'[A-Z]', cur_word[0])):
                self.insert_key(self.dict_key)

    def get_feature_index_from_history(self, history: History):
        if bool(re.search(r'[A-Z]', history.cword[0])):
            if self.dict.get(self.dict_key, None):
                return 0, self.dict[self.dict_key]
            else:
                return -1, -1
        else:
            return -1, -1


class HasAllCapitalLettersDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'all_capital_letters'

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Extract out of text all words that contain capital letters - <w_i | w_i contains capital letter>
            fill all words with capital letters with index of appearance
        """
        for i in range(len(word_ordered_list)):
            cur_word = word_ordered_list[i]
            if cur_word.upper() == cur_word:
                self.insert_key(self.dict_key)

    def get_feature_index_from_history(self, history: History):
        if history.cword == history.cword.upper():
            if self.dict.get(self.dict_key, None):
                return 0, self.dict[self.dict_key]
            else:
                return -1, -1
        else:
            return -1, -1


class HasDigitDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'digit'

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Extract out of text all words that contain digit - <w_i | w_i contains digit>
            fill all words with digit with index of appearance
        """
        for i in range(len(word_ordered_list)):
            cur_word = word_ordered_list[i]
            if bool(re.search(r'\d', cur_word)):
                self.insert_key(self.dict_key)

    def get_feature_index_from_history(self, history: History):
        if bool(re.search(r'\d', history.cword)):
            if self.dict.get(self.dict_key, None):
                return 0, self.dict[self.dict_key]
            else:
                return -1, -1
        else:
            return -1, -1


class HasSameTagDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'same_tag'

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Check if the word is in the dict of words that have same tag most of the times across all training set
        """
        for i in range(len(word_ordered_list)):
            cur_word = word_ordered_list[i]
            if common_tags.get(cur_word, None):
                self.insert_key(self.dict_key)

    def get_feature_index_from_history(self, history: History):
        if common_tags.get(history.cword, None):
            if self.dict.get(self.dict_key, None):
                return 0, self.dict[self.dict_key]
            else:
                return -1, -1
        else:
            return -1, -1


class ContainsLetterDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'contains_letter'

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Check if the word has letters
        """
        for i in range(len(word_ordered_list)):
            cur_word = word_ordered_list[i]
            if bool(re.search(r'[a-zA-Z]', cur_word)):
                self.insert_key(self.dict_key)

    def get_feature_index_from_history(self, history: History):
        if bool(re.search(r'[a-zA-Z]', history.cword)):
            if self.dict.get(self.dict_key, None):
                return 0, self.dict[self.dict_key]
            else:
                return -1, -1
        else:
            return -1, -1


class ContainsHyphenDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'contains_hyphen'

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Check if the word has letters
        """
        for i in range(len(word_ordered_list)):
            cur_word = word_ordered_list[i]
            if bool(re.search(r'-', cur_word)):
                self.insert_key(self.dict_key)

    def get_feature_index_from_history(self, history: History):
        if bool(re.search(r'-', history.cword)):
            if self.dict.get(self.dict_key, None):
                return 0, self.dict[self.dict_key]
            else:
                return -1, -1
        else:
            return -1, -1


class IsFirstWordDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'is_first'

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Check if the word is first in sentence
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |,\n', line)

                if len(splited_words) == 1:
                    continue
                del splited_words[-1]

                cur_word, _ = re.split('_', splited_words[0])
                self.insert_key(self.dict_key)

    def get_feature_index_from_history(self, history: History):
        if history.pword == '*':
            # check if was not filtered
            if self.dict.get(self.dict_key, None):
                return 0, self.dict[self.dict_key]
            else:
                return -1, -1
        else:
            return -1, -1


class IsLastWordDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'is_last'

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Check if the word is first in sentence
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |,\n', line)
                if len(splited_words) == 1:
                    continue
                del splited_words[-1]

                cur_word, _ = re.split('_', splited_words[-1])
                self.insert_key(self.dict_key)

    def get_feature_index_from_history(self, history: History):
        if history.nword == '*':
            # check if was not filtered
            if self.dict.get(self.dict_key, None):
                return 0, self.dict[self.dict_key]
            else:
                return -1, -1
        else:
            return -1, -1


class HasSameTagAlwaysDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_keys = sorted([attr for attr in dir(WordConstantTagSets) if attr.endswith('set')])
        self.sets_dict = OrderedDict()
        self.init_dict()

    def init_dict(self):
        for key in self.dict_keys:
            self.sets_dict[key] = getattr(WordConstantTagSets, key)

    def fill_dict(self, word_ordered_list, tag_ordered_list, file_path):
        """
            Check if the word is in the dict of words that have same tag across all training set
        """
        for i in range(len(word_ordered_list)):
            cur_word = word_ordered_list[i]
            for idx, k in enumerate(self.dict_keys):
                if cur_word in self.sets_dict[k]:
                    self.insert_key(k)
                    break

    def get_feature_index_from_history(self, history: History):
        for idx, k in enumerate(self.dict_keys):
            if history.cword in self.sets_dict[k]:
                # check if was not filtered
                if self.dict.get(k, None):
                    return idx, self.dict[k]
                else:
                    return -1, -1
        return -1, -1
