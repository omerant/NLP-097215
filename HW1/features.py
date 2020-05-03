from collections import OrderedDict, namedtuple
from abc import abstractmethod, ABC
import re
import utils
from utils import History, Symbols


class WordAndTagConstants:
    PTAG_SENTENCE_BEGINNING = '*'
    PPTAG_SENTENCE_BEGINNING = '**'
    PWORD_SENTENCE_BEGINNING = '&&'
    NWORD_SENTENCE_END = '&&&'
    PPWORD_SENTENCE_BEGINNING = '^^'
    NNWORD_SENTENCE_END = '^^^'


class FeatureDict(ABC):
    INVALID_IDX = -1
    INVALID_VAL = -1

    def __init__(self):
        self.dict = OrderedDict()

    @abstractmethod
    def fill_dict(self, hist_sentence_list: [[History]]):
        pass

    @abstractmethod
    def get_feature_index_and_count_from_history(self, history: History) -> (int, int):
        pass

    def insert_key(self, key):
        if key not in self.dict.keys():
            self.dict[key] = 1
        else:
            self.dict[key] += 1

    def get_key_index(self, key):
        if key not in self.dict.keys():
            return self.INVALID_IDX, self.INVALID_VAL

        for idx, k in enumerate(self.dict.keys()):
            if key == k:
                return idx, self.dict[key]


class TrigramTagsCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text ordered tag triplets - <t_i-2, t_i-1, t_i>
            fill all ordered tag triplets with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_triplet = (hist.pptag, hist.ptag, hist.ctag)
                self.insert_key(cur_triplet)

    def get_feature_index_and_count_from_history(self, history: History):
        key = (history.pptag, history.ptag, history.ctag)
        return self.get_key_index(key)


class BigramTagsCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text ordered tag pairs - <t_i-1, t_i>
            fill all ordered tag pairs with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_pair = (hist.ptag, hist.ctag)
                self.insert_key(cur_pair)

    def get_feature_index_and_count_from_history(self, history: History):
        key = (history.ptag, history.ctag)
        return self.get_key_index(key)


class UnigramTagsCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text tags - <t_i>
            fill all tags with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_tag = hist.ctag
                self.insert_key(cur_tag)

    def get_feature_index_and_count_from_history(self, history: History):
        key = history.ctag
        return self.get_key_index(key)


class WordsTagsCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text all word/tag pairs <w_i, t_i>
            fill all word/tag pairs with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                cur_tag = hist.ctag
                self.insert_key((cur_word, cur_tag))

    def get_feature_index_and_count_from_history(self, history: History):
        key = (history.cword, history.ctag)
        return self.get_key_index(key)


class WordsPrefixTagsCountDict(FeatureDict):
    def __init__(self, pref_len):
        super().__init__()
        self.pref_len = pref_len

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text all word_prefix/tag pairs for word_prefix| <=4 - <pref_w_i, t_i>
            fill all word_prefix/tag pairs with index of appearance

        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                cur_tag = hist.ctag
                if len(cur_word) < self.pref_len:
                    continue
                pref = cur_word[:self.pref_len]
                self.insert_key((pref, cur_tag))

    def get_feature_index_and_count_from_history(self, history: History):
        cur_word = history.cword
        cur_tag = history.ctag
        if len(cur_word) < self.pref_len:
            return self.INVALID_IDX, self.INVALID_VAL
        cur_pref = cur_word[:self.pref_len]
        key = (cur_pref, cur_tag)
        return self.get_key_index(key)


class WordsSuffixTagsCountDict(FeatureDict):
    def __init__(self, suff_len=1):
        super().__init__()
        self.suff_len = suff_len

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text all word_suffix/tag pairs for |word_suffix| <=4 - <suff_w_i, t_i>
            fill all word_suffix/tag pairs with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                cur_tag = hist.ctag
                if len(cur_word) < self.suff_len:
                    continue
                pref = cur_word[-self.suff_len:]
                self.insert_key((pref, cur_tag))

    def get_feature_index_and_count_from_history(self, history: History):
        cur_word = history.cword
        cur_tag = history.ctag
        if len(cur_word) < self.suff_len:
            return self.INVALID_IDX, self.INVALID_VAL
        cur_suff = cur_word[-self.suff_len:]
        key = (cur_suff, cur_tag)
        return self.get_key_index(key)


class PrevWordCurrTagCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text all word/tag pairs - <w_i-1, t_i>
            fill all word/tag pairs with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_tag = hist.ctag
                prev_word = hist.pword
                self.insert_key((prev_word, cur_tag))

    def get_feature_index_and_count_from_history(self, history: History):
        key = (history.pword, history.ctag)
        return self.get_key_index(key)

class DoublePrevWordCurrTagCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text all word/tag pairs - <w_i-2, t_i>
            fill all word/tag pairs with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_tag = hist.ctag
                double_prev_word = hist.ppword
                self.insert_key((double_prev_word, cur_tag))

    def get_feature_index_and_count_from_history(self, history: History):
        key = (history.ppword, history.ctag)
        return self.get_key_index(key)

class NextWordCurrTagCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text all word/tag pairs - <w_i+1, t_i>
            fill all word/tag pairs with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_tag = hist.ctag
                next_word = hist.nword
                self.insert_key((next_word, cur_tag))

    def get_feature_index_and_count_from_history(self, history: History):
        key = (history.nword, history.ctag)
        return self.get_key_index(key)

class DoubleNextWordCurrTagCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text all word/tag pairs - <w_i+2, t_i>
            fill all word/tag pairs with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_tag = hist.ctag
                double_next_word = hist.nnword
                self.insert_key((double_next_word, cur_tag))

    def get_feature_index_and_count_from_history(self, history: History):
        key = (history.nnword, history.ctag)
        return self.get_key_index(key)


class SkipBigramCountDict(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text ordered tag skip bigrams - <t_i-2, t_i>
            fill all ordered tag skip bigrams with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_tag = hist.ctag
                pptag = hist.pptag
                self.insert_key((pptag, cur_tag))

    def get_feature_index_and_count_from_history(self, history: History):
        key = (history.pptag, history.ptag, history.ctag)
        return self.get_key_index(key)


class HasFirstCapitalLetterDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'begins_with_capital_letter'

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text all words that contain first capital letters - <w_i | w_i has first capital letter>
            fill all words with capital letters with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                if bool(re.search(r'[A-Z]', cur_word[0])):
                    self.insert_key(self.dict_key)

    def get_feature_index_and_count_from_history(self, history: History):
        if bool(re.search(r'[A-Z]', history.cword[0])):
            return 0, self.dict.get(self.dict_key, 0)
        else:
            return self.INVALID_IDX, self.INVALID_VAL


class HasAllCapitalLettersDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'all_capital_letters'

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text all words that contain capital letters - <w_i | w_i contains capital letter>
            fill all words with capital letters with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                if cur_word.upper() == cur_word:
                    self.insert_key(self.dict_key)

    def get_feature_index_and_count_from_history(self, history: History):
        if history.cword == history.cword.upper():
            return 0, self.dict.get(self.dict_key, 0)
        else:
            return self.INVALID_IDX, self.INVALID_VAL


class HasDigitDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'digit'

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text all words that contain digit - <w_i | w_i contains digit>
            fill all words with digit with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                if bool(re.search(r'\d', cur_word)):
                    self.insert_key(self.dict_key)

    def get_feature_index_and_count_from_history(self, history: History):
        if bool(re.search(r'\d', history.cword)):
            return 0, self.dict.get(self.dict_key, 0)
        else:
            return self.INVALID_IDX, self.INVALID_VAL


class HasOnlyDigitDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'contains_only_digits'

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text all words have only digits - <w_i | w_i has only digits>
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                if utils.is_number(cur_word):
                    self.insert_key(self.dict_key)

    def get_feature_index_and_count_from_history(self, history: History):
        if history.cword.isdigit():
            return 0, self.dict.get(self.dict_key, 0)
        else:
            return self.INVALID_IDX, self.INVALID_VAL


class ContainsLetterDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'contains_letter'

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Check if the word has letters
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                if bool(re.search(r'[a-zA-Z]', cur_word)):
                    self.insert_key(self.dict_key)

    def get_feature_index_and_count_from_history(self, history: History):
        if bool(re.search(r'[a-zA-Z]', history.cword)):
            return 0, self.dict.get(self.dict_key, 0)
        else:
            return self.INVALID_IDX, self.INVALID_VAL


class ContainsOnlyLettersDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'contains_only_letters'

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Check if the word has letters
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                if cur_word.isalpha():
                    self.insert_key(self.dict_key)

    def get_feature_index_and_count_from_history(self, history: History):
        if history.cword.isalpha():
            return 0, self.dict.get(self.dict_key, 0)
        else:
            return self.INVALID_IDX, self.INVALID_VAL


class ContainsHyphenDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'contains_hyphen'

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Check if the word has letters
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                if bool(re.search(r'-', cur_word)):
                    self.insert_key(self.dict_key)

    def get_feature_index_and_count_from_history(self, history: History):
        if bool(re.search(r'-', history.cword)):
            return 0, self.dict.get(self.dict_key, 0)
        else:
            return self.INVALID_IDX, self.INVALID_VAL


class IsFirstWordDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'is_first'

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Check if the word is first in sentence
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                pword = hist.pword
                if pword == WordAndTagConstants.PWORD_SENTENCE_BEGINNING:
                    self.insert_key(self.dict_key)

    def get_feature_index_and_count_from_history(self, history: History):
        if history.pword == WordAndTagConstants.PWORD_SENTENCE_BEGINNING:
            return 0, self.dict.get(self.dict_key, 0)
        else:
            return self.INVALID_IDX, self.INVALID_VAL


class IsLastWordDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'is_last'

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Check if the word is last in sentence
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                nword = hist.nword
                if nword == WordAndTagConstants.NWORD_SENTENCE_END:
                    self.insert_key(self.dict_key)

    def get_feature_index_and_count_from_history(self, history: History):
        if history.nword == WordAndTagConstants.NWORD_SENTENCE_END:
            return 0, self.dict.get(self.dict_key, 0)
        else:
            return self.INVALID_IDX, self.INVALID_VAL


class ContainsSymbolDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'contains_symbol'

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Check if the word contains a symbol
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                for letter in cur_word:
                    if letter in Symbols:
                        self.insert_key(self.dict_key)

    def get_feature_index_and_count_from_history(self, history: History):
        cur_word = history.cword
        for letter in cur_word:
            if letter in Symbols:
                return 0, self.dict.get(self.dict_key, 0)
        else:
            return self.INVALID_IDX, self.INVALID_VAL


class ContainsOnlySymbolsDict(FeatureDict):
    def __init__(self):
        super().__init__()
        self.dict_key = 'contains_symbol'

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Check if the word contains a symbol
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                all_symbols = True
                for letter in cur_word:
                    if letter not in Symbols:
                        all_symbols = False
                if all_symbols:
                    self.insert_key(self.dict_key)

    def get_feature_index_and_count_from_history(self, history: History):
        cur_word = history.cword
        for letter in cur_word:
            if letter not in Symbols:
                return self.INVALID_IDX, self.INVALID_VAL

        return 0, self.dict.get(self.dict_key, 0)


class WordsLengthDict(FeatureDict):
    def __init__(self, len):
        super().__init__()
        self.len = len
        self.dict_key = 'word_length'

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text all word length/tag pairs - <len(w), t_i>
            fill all word len/tag pairs with index of appearance

        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_word = hist.cword
                cur_tag = hist.ctag
                if len(cur_word) == self.len:
                    self.insert_key((self.len))

    def get_feature_index_and_count_from_history(self, history: History):
        cur_word = history.cword
        if len(cur_word) != self.len:
            return self.INVALID_IDX, self.INVALID_VAL
        return 0, self.dict.get(self.dict_key, 0)


class TwoPreviousTagsAndCurrentWord(FeatureDict):
    def __init__(self):
        super().__init__()

    def fill_dict(self, hist_sentence_list: [[History]]):
        """
            Extract out of text ordered two previous tags and current word - <t_i-2, t_i-1, w_i>
            fill all ordered tag triplets with index of appearance
        """
        for sentence in hist_sentence_list:
            for hist in sentence:
                cur_triplet = (hist.pptag, hist.ptag, hist.cword)
                self.insert_key(cur_triplet)

    def get_feature_index_and_count_from_history(self, history: History):
        key = (history.pptag, history.ptag, history.cword)
        return self.get_key_index(key)

