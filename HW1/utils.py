import os
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
import time
import re
from collections import namedtuple

MIN_EXP_VAL = -100
MIN_LOG_VAL = 1/(10**60)
BASE_PROB = 1/(10**19)

History = namedtuple('History', 'cword, pptag, ptag, ctag, nword, pword, nnword, ppword')
Symbols = """!#$%&?/\|}{~:;.,'`-]"""
STR_CHECK_SYMBOL = re.compile(Symbols)
UNKNOWN_WORD = 'UNK'


class OpenClassTypes:
    VERB = {'VB', 'VBD', 'VBG', 'VBN'}
    ADVERB = {'RB', 'RBR', 'RBS'}
    ADJECTIVE = {'JJ', 'JJR', 'JJS'}
    NOUN = {'NN', 'NNS', 'NNP', 'NNPS'}


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed


def is_number(s):
    return bool(re.match(r'^-?\d+(?:\,\d+)?$', s)) or bool(re.match(r'^-?\d+(?:\.\d+)?$', s))
