import numpy as np
import os
import pickle
from pre_processing import FeatureStatistics
from features import History
from scipy.optimize import fmin_l_bfgs_b

np.seterr(all='raise')


class MaximumEntropyMarkovModel:
    def __init__(self, train_data_path, threshold, reg_lambda):
        self.feature_statistics = FeatureStatistics(train_data_path, threshold)
        self.feature_statistics.pre_process(False)
        self.dump_weights_path = 'weights'
        self.reg_lambda = reg_lambda
        self.prob_dict = dict()

    @staticmethod
    def load_v_from_pickle(dump_weights_path, threshold, reg_lambda):
        weights_dir = os.path.join(dump_weights_path, str(threshold) +
                                   f'-reg_lambda={reg_lambda}')

        with open(weights_dir, 'rb') as f:
            v = pickle.load(f).reshape(-1)
        return v

    def load_prob_dict(self):
        print('loading probability dict')
        dump_path = os.path.join('probability_dict',
                                 f'version={self.feature_statistics.version}_threshold={self.feature_statistics.threshold}')
        with open(dump_path, 'rb') as f:
            self.prob_func = pickle.load(f)

        print('finished loading probability dict')

    def calc_empirical_counts(self):
        features_linear_term_indices = [self.feature_statistics.all_possible_tags_dict[hist] for sentence in
                                        self.feature_statistics.history_sentence_list for hist in sentence]
        empirical_counts = np.zeros(self.feature_statistics.num_features, dtype=np.float32)
        for feature_ind in features_linear_term_indices:
            empirical_counts[feature_ind] += 1

        return empirical_counts

    @staticmethod
    def calc_normalization_term_exp_dict_prob_dict(v, all_possible_hist_feature_dict, sentence_history_list,
                                                   word_to_tags_set_dict, word_to_most_probable_tag_set):
        norm_term = 0.
        exp_dict = dict()
        prob_dict = dict()
        for sentence in sentence_history_list:
            for hist in sentence:
                norm_i = 0.
                cur_possible_hist_list = []
                if word_to_most_probable_tag_set.get(hist.cword, None):
                    tag_set = {word_to_most_probable_tag_set[hist.cword][0]}
                else:
                    tag_set = word_to_tags_set_dict[hist.cword]

                # fill exp dict
                for tag in tag_set:
                    n_hist = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag,
                                     nword=hist.nword, pword=hist.pword, ctag=tag)

                    if not exp_dict.get(n_hist, None):
                        dot_prod = np.sum(v[all_possible_hist_feature_dict[n_hist]])
                        exp_dict[n_hist] = np.exp(dot_prod).astype(np.float128)

                    cur_possible_hist_list.append(n_hist)
                    norm_i += exp_dict[n_hist]

                # fill prob_dict
                for idx, hist in enumerate(cur_possible_hist_list):
                    if len(cur_possible_hist_list) == 1:
                        prob_dict[hist] = 1
                    else:
                        prob_dict[hist] = np.float128(exp_dict[hist] / norm_i)

                # update normzliaztion term
                if len(cur_possible_hist_list) == 1:
                    norm_term += np.sum(v[all_possible_hist_feature_dict[n_hist]]) #dot prod
                else:
                    norm_term += np.log(norm_i)

        return norm_term, prob_dict, exp_dict

    @staticmethod
    def calc_expected_counts(history_sentence_list, prob_dict, all_possible_hist_feature_dict,
                             word_to_tags_set_dict, word_to_most_probable_tag_set, v):
        expected_counts = np.zeros(v.shape, np.float32)
        for sentence in history_sentence_list:
            for hist in sentence:
                if word_to_most_probable_tag_set.get(hist.cword, None):
                    tag_set = {word_to_most_probable_tag_set[hist.cword][0]}
                else:
                    tag_set = word_to_tags_set_dict[hist.cword]
                for tag in tag_set:
                    n_hist = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag,
                                     ctag=tag, nword=hist.nword, pword=hist.pword)
                    expected_counts[all_possible_hist_feature_dict[n_hist]] += 1 * prob_dict[n_hist]

        return expected_counts

    @staticmethod
    def calc_objective_per_iter(*args):
        v = args[0]
        all_possible_hist_feature_dict = args[1]
        reg_lambda = args[2]
        empirical_counts = args[3]
        sentence_history_list = args[4]
        word_to_tags_set_dict = args[5]
        word_to_most_probable_tag_set = args[6]
        linear_term = v.dot(empirical_counts)

        normalization_term, prob_dict, _ = MaximumEntropyMarkovModel.calc_normalization_term_exp_dict_prob_dict(
            v, all_possible_hist_feature_dict, sentence_history_list,
            word_to_tags_set_dict, word_to_most_probable_tag_set
        )

        regularization_term = 0.5 * reg_lambda * np.linalg.norm(v, ord=2)
        likelihood = linear_term - normalization_term - regularization_term
        # FINISHED CALCULATING LIKELIHOOD
        expected_counts = MaximumEntropyMarkovModel.calc_expected_counts(
            sentence_history_list, prob_dict, all_possible_hist_feature_dict, word_to_tags_set_dict,
            word_to_most_probable_tag_set, v
        )
        regularization_grad = reg_lambda * v
        grad = empirical_counts - expected_counts - regularization_grad
        # FINISHED CALCULATING GRAD
        return (-1) * likelihood, (-1) * grad

    def optimize_model(self):
        arg_1 = self.feature_statistics.all_possible_tags_dict
        arg_2 = self.reg_lambda
        args_3 = self.calc_empirical_counts()
        args_4 = self.feature_statistics.history_sentence_list
        args_5 = self.feature_statistics.word_possible_tag_set
        args_6 = self.feature_statistics.word_possible_tag_with_threshold_dict
        args = (arg_1, arg_2, args_3, args_4, args_5, args_6)
        w_0 = np.random.normal(0, 0.01, (self.feature_statistics.num_features)).astype(np.float128)
        # w_0 = np.zeros(self.feature_statistics.num_features, dtype=np.float32)
        optimal_params = fmin_l_bfgs_b(func=self.calc_objective_per_iter, x0=w_0, args=args, maxiter=10000, iprint=1)
        weights = optimal_params[0]
        print(weights)
        weights_dir = os.path.join(self.dump_weights_path, str(self.feature_statistics.threshold) +
                                   f'-reg_lambda={arg_2}')

        self.v = weights
        with open(weights_dir, 'wb') as f:
            pickle.dump(weights, f)