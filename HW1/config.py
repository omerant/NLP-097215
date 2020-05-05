import features as ft


class Cfg:
    feature_dicts = {}


class BaseCfg(Cfg):
    feature_dicts = {
        # Init all features dictionaries - each feature dict's name should start with fd
        # fill dict for 1-3 ngrams
        ft.TrigramTagsCountDict,
        ft.BigramTagsCountDict,
        ft.UnigramTagsCountDict,
        ft.WordsTagsCountDict,
        ft.WordsPrefixTagsCountDict,
        ft.WordsSuffixTagsCountDict,
        ft.PrevWordCurrTagCountDict,
        ft.NextWordCurrTagCountDict,
        ft.DoublePrevWordCurrTagCountDict,
        ft.DoubleNextWordCurrTagCountDict,
        ft.SkipBigramCountDict,
        # letters digits
        ft.HasFirstCapitalLetterDict,
        ft.HasAllCapitalLettersDict,
        ft.HasDigitDict,
        ft.HasOnlyDigitDict,
        ft.ContainsHyphenDict,
        ft.IsFirstWordDict,
        ft.ContainsSymbolDict,
        ft.ContainsOnlySymbolsDict,
        ft.TwoPreviousTagsAndCurrentWord,
    }
