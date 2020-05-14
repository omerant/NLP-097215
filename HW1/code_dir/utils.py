import time
import re
from collections import namedtuple


History = namedtuple('History', 'cword, pptag, ptag, ctag, nword, pword, nnword, ppword')
Symbols = """!#$%&?/\|}{~:;.,'`-]"""


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
