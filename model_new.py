import numpy as np
import os
import pickle
from pre_processing import FeatureStatistics
from features import History
from scipy.optimize import fmin_l_bfgs_b
from viterbi import Viterbi


class MaximumEntropyMarkovModel:
    def __init__(self, train_data_path, test_data_path):
        self.feature_statistics = FeatureStatistics(train_data_path)
        self.feature_statistics.pre_process()
        self.test_data = FeatureStatistics(test_data_path)
        if test_data_path:
            self.test_data.fill_ordered_history_list()
        self.dump_weights_path = 'weights'
        self.prob_dict = dict()
        self.v = None
        # self.v[1] = 0.01# start with uniform distribution
        # self.prob_func = self.load_prob_dict()
        # self.v = self._load_v_from_pickle()

    def load_v_from_pickle(self):
        print('filling probability dict')
        weights_dir = os.path.join(self.dump_weights_path, str(self.feature_statistics.version) +
                                   f'-reg_lambda={0.1}-num_iter={1}')
        with open(weights_dir, 'rb') as f:
            self.v = pickle.load(f).reshape(-1)

    def fill_prob_dict(self):
        print('filling probability dict')
        tag_set = self.feature_statistics.tags_set - {'*'}
        _, prob_dict = self.calc_normalization_term_exp_dict_prob_dict(self.v,
                                                                        self.feature_statistics.all_possible_tags_dict,
                                                                        self.feature_statistics.history_ordered_list,
                                                                        tag_set
                                                        )
        self.prob_dict = prob_dict
        print('finished filling probability dict')

    def load_prob_dict(self):
        print('loading probability dict')
        dump_path = os.path.join('probability_dict',
                                 f'version={self.feature_statistics.version}_threshold={self.feature_statistics.threshold}')
        with open(dump_path, 'rb') as f:
            self.prob_func = pickle.load(f)

        print('finished loading probability dict')

    def calc_empirical_counts(self):
        print('----calculating empirical counts-----')
        features_linear_term_indices = [self.feature_statistics.all_possible_tags_dict[hist] for hist in
                                        self.feature_statistics.history_ordered_list]
        print(f'----finished getting features len_all_feature={len(features_linear_term_indices)}-----')
        empirical_counts = np.zeros(self.feature_statistics.num_features, dtype=np.float64)
        for feature_ind in features_linear_term_indices:
            empirical_counts[feature_ind] += 1

        dump_path = os.path.join('empirical_counts', f'version={self.feature_statistics.version}_threshold={self.feature_statistics.threshold}')
        with open(dump_path, 'wb') as f:
            pickle.dump(empirical_counts, f)
        print('----finished calculating empirical counts-----')
        return empirical_counts

    @staticmethod
    def calc_normalization_term_exp_dict_prob_dict(v, all_possible_hist_feature_dict, history_list, tag_set):
        norm_term = 0.
        exp_dict = dict()
        prob_dict = dict()
        for hist in history_list:
            norm_i = 0.
            cur_possible_hist_list = []
            # fill exp dict
            for tag in tag_set:
                n_hist = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag,
                                 nword=hist.nword, pword=hist.pword, ctag=tag)

                if not exp_dict.get(n_hist, None):
                    dot_prod = np.sum(v[all_possible_hist_feature_dict[n_hist]])
                    exp_dict[n_hist] = np.exp(dot_prod)
                    cur_possible_hist_list.append(n_hist)

                norm_i += exp_dict[n_hist]
            # fill prob_dict
            for hist in cur_possible_hist_list:
                prob_dict[hist] = exp_dict[hist] / norm_i

            # update normzliaztion term
            norm_term += np.log(norm_i)

        return norm_term, prob_dict

    @staticmethod
    def calc_expected_counts(history_list, prob_dict, all_possible_hist_feature_dict, tag_set, v):
        expected_counts = np.zeros(v.shape, np.float64)
        for hist in history_list:
            for tag in tag_set:
                n_hist = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag,
                                 ctag=tag, nword=hist.nword, pword=hist.pword)
                expected_counts[all_possible_hist_feature_dict[n_hist]] += 1 * prob_dict[n_hist]
        return expected_counts

    @staticmethod
    def calc_objective_per_iter(*args):
        v = args[0]
        print(v)
        all_possible_hist_feature_dict = args[1]
        reg_lambda = args[2]
        empirical_counts = args[3]
        history_list = args[4]
        tag_set = args[5] - {'*'}
        linear_term = v.dot(empirical_counts)

        normalization_term, prob_dict = MaximumEntropyMarkovModel.calc_normalization_term_exp_dict_prob_dict(
            v, all_possible_hist_feature_dict, history_list, tag_set
        )

        regularization_term = 0.5 * reg_lambda * np.linalg.norm(v, ord=2)
        likelihood = linear_term - normalization_term - regularization_term
        # FINISHED CALCULATING LIKELIHOOD
        expected_counts = MaximumEntropyMarkovModel.calc_expected_counts(
            history_list, prob_dict, all_possible_hist_feature_dict, tag_set, v
        )
        regularization_grad = reg_lambda * v
        grad = empirical_counts - expected_counts - regularization_grad
        # FINISHED CALCULATING GRAD
        print(f'likelihood: {likelihood}')
        return (-1) * likelihood, (-1) * grad

    def optimize_model(self):
        arg_1 = self.feature_statistics.all_possible_tags_dict
        arg_2 = 0.1  # lambda_reg
        args_3 = self.calc_empirical_counts()
        args_4 = self.feature_statistics.history_ordered_list
        args_5 = self.feature_statistics.tags_set
        args = (arg_1, arg_2, args_3, args_4, args_5)
        w_0 = np.zeros(self.feature_statistics.num_features, dtype=np.float64)
        optimal_params = fmin_l_bfgs_b(func=self.calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=1)
        weights = optimal_params[0]
        print(weights)
        weights_dir = os.path.join(self.dump_weights_path, str(self.feature_statistics.version) +
                                   f'-reg_lambda={arg_2}-num_iter={1}')
        self.v = weights
        with open(weights_dir, 'wb') as f:
            pickle.dump(weights, f)


if __name__ == '__main__':
    train1_path = 'data/train1.wtag'
    test1_short_path = 'data/test1.wtag'
    import timeit
    start = timeit.default_timer()
    memm = MaximumEntropyMarkovModel(train_data_path=train1_path, test_data_path=test1_short_path)
    # memm.optimize_model()
    memm.load_v_from_pickle()
    memm.fill_prob_dict()
    viterbi = Viterbi(
        v=memm.v,
        sentence_hist_list=memm.test_data.history_sentence_list,
        tags_set=memm.feature_statistics.tags_set,
        all_possible_tags_dict=memm.feature_statistics.all_possible_tags_dict,
        get_feature_from_hist=memm.feature_statistics.get_non_zero_sparse_feature_vec_indices_from_history
    )
    all_res_list, all_acc_list = viterbi.predict_all()
    # print(f'avg acc: {sum(all_acc_list)/len(all_acc_list)}')
    stop = timeit.default_timer()
    print(f'execution time: {stop - start}')
