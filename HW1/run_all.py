import argparse
from pre_processing import FeatureStatistics
from model_new import MaximumEntropyMarkovModel
from viterbi import Viterbi
from utils import timeit
from config import BaseCfg, NoPrefSufCfg


parser = argparse.ArgumentParser()

parser.add_argument("--threshold", help="threshold that will be used to filter features", type=int, required=True)
parser.add_argument("--train-path", help="path to training data", type=str, required=True)
parser.add_argument("--test-path", help="path to test data", type=str, required=True)
parser.add_argument("--reg-lambda", help="regularization lambda", type=float, required=True)
parser.add_argument("--beam", help="beam width for viterbi", type=int, required=True)
parser.add_argument("--config", help="regularization lambda", choices=['base', 'nps', 'comp1', 'comp2'], required=True)

parser.add_argument("--run-all", help="run pre_process + train + predict", type=bool, default=False)
parser.add_argument("--pp", help="run only pre_process", type=bool, required=False, default=False)
parser.add_argument("--tr", help="run only train", type=bool, required=False, default=False)
parser.add_argument("--pr", help="run only predict", type=bool, required=False, default=False)
args = parser.parse_args()


@timeit
def pre_process(train_path, threshold, config):
    feature_statistics = FeatureStatistics(input_file_path=train_path, threshold=threshold, config=config)
    feature_statistics.pre_process(fill_possible_tag_dict=True)


@timeit
def train(train_path, threshold, reg_lambda, conf):
    memm = MaximumEntropyMarkovModel(train_data_path=train_path, threshold=threshold,
                                     reg_lambda=reg_lambda, config=conf)
    memm.optimize_model()


@timeit
def predict(train_path, threshold, reg_lambda, test_path, conf, beam_width):
    v = MaximumEntropyMarkovModel.load_v_from_pickle(dump_weights_path='weights', threshold=args.threshold,
                                                     reg_lambda=reg_lambda)
    ft_statistics = FeatureStatistics(input_file_path=train_path, threshold=threshold, config=conf)
    ft_statistics.pre_process(fill_possible_tag_dict=False)
    test_sentence_hist_list = FeatureStatistics.fill_ordered_history_list(file_path=test_path, is_test=True)
    tag_set = ft_statistics.tags_list
    all_possible_tags_dict = ft_statistics.hist_to_feature_vec_dict
    get_ft_from_hist_func = ft_statistics.get_non_zero_feature_vec_indices_from_history
    word_possible_tag_set = ft_statistics.word_possible_tag_set
    word_possible_tag_with_threshold_dict = ft_statistics.word_possible_tag_with_threshold_dict
    rare_words_tags = ft_statistics.rare_words_tags

    viterbi = Viterbi(
        v=v, sentence_hist_list=test_sentence_hist_list, tags_list=tag_set,
        all_possible_tags_dict=all_possible_tags_dict, get_feature_from_hist=get_ft_from_hist_func,
        word_possible_tag_set=word_possible_tag_set,
        word_possible_tag_with_threshold_dict=word_possible_tag_with_threshold_dict,
        rare_words_tags=rare_words_tags,
        threshold=args.threshold,
        reg_lambda=args.reg_lambda,
        beam_width=beam_width
    )
    viterbi.predict_all_test()


def run_all(train_path, threshold, reg_lambda, test_path):
    pre_process(train_path=train_path, threshold=threshold)
    train(train_path=train_path, threshold=threshold, reg_lambda=reg_lambda)
    predict(train_path=train_path, threshold=threshold, reg_lambda=reg_lambda, test_path=test_path)
# run example:
# python run_all.py --th 10 --tra data/train1.wtag --te data/test1.wtag --reg-lambda 0.01 --run-all true --conf base --beam 6
# pre_process  only
# python run_all.py --th 10 --tra data/train1.wtag --te data/test1_short.wtag --reg-lambda 0.01 --pp true --conf base --beam 6
# train only
# python run_all.py --th 10 --tra data/train1.wtag --te data/test1_short.wtag --reg-lambda 0.01 --tr true --conf base --beam 6
# predict only
# python run_all.py --th 10 --tra data/train1.wtag --te data/test1_short.wtag --reg-lambda 0.01 --pr true --conf base --beam 6


if __name__ == '__main__':
    config = None
    if args.config == 'base':
        config = BaseCfg
    elif args.config == 'nps':
        config = NoPrefSufCfg
    # TODO: add another configurations when needed
    if args.run_all:
        print('RUNNING ALL FLOW')
        pre_process(train_path=args.train_path, threshold=args.threshold, config=config)
        train(train_path=args.train_path, threshold=args.threshold, reg_lambda=args.reg_lambda, conf=config)
        predict(train_path=args.train_path, threshold=args.threshold, reg_lambda=args.reg_lambda,
                test_path=args.test_path, conf=config, beam_width=args.beam)
    elif args.pp:
        print('RUNNING ONLY PRE PROCESS')
        pre_process(train_path=args.train_path, threshold=args.threshold, config=config)
    elif args.tr:
        print('RUNNING ONLY TRAINING')
        train(train_path=args.train_path, threshold=args.threshold, reg_lambda=args.reg_lambda, conf=config)
    elif args.pr:
        print('RUNNING ONLY PREDICT')
        predict(train_path=args.train_path, threshold=args.threshold, reg_lambda=args.reg_lambda,
                test_path=args.test_path, conf=config, beam_width=args.beam)
