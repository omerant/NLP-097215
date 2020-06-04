from collections import defaultdict

WORD_IDX = 1
POS_IDX = 3
HEAD_IDX = 6
IGNORE_IDX = -20


def split(string, delimiters):
    """
        Split strings according to delimiters
        :param string: full sentence
        :param delimiters string: characters for spliting
            function splits sentence to words
    """
    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)

    return stack


def get_vocabs_dep_parser(list_of_paths):
    """
        Extract vocabs from given datasets. Return a word2ids and tag2idx.
        :param file_paths: a list with a full path for all corpuses
            Return:
              - word2idx
              - tag2idx
    """
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                if line == '\n':
                    continue
                splited_words = split(line, (' ', '\n', '\t'))
                word_dict[splited_words[WORD_IDX]] += 1
                pos_dict[splited_words[POS_IDX]] += 1

    return word_dict, pos_dict


def get_vocabs(list_of_paths):
    """
        Extract vocabs from given datasets. Return a word2ids and tag2idx.
        :param file_paths: a list with a full path for all corpuses
            Return:
              - word2idx
              - tag2idx
    """
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                splited_words = split(line, (' ', '\n'))
                del splited_words[-1]
                for word_and_tag in splited_words:
                    word, pos_tag = split(word_and_tag, '_')
                    word_dict[word] += 1
                    pos_dict[pos_tag] += 1

    return word_dict, pos_dict

