import numpy as np
import os
import pickle
from pre_processing import FeatureStatistics
from utils import History
from scipy.optimize import fmin_l_bfgs_b
from utils import MIN_EXP_VAL, MIN_LOG_VAL, BASE_PROB
import timeit
from scipy.sparse import csr_matrix
np.random.seed(0)
np.seterr(all='raise')


class MaximumEntropyMarkovModel:
    def __init__(self, train_data_path, threshold, reg_lambda, config):
        self.feature_statistics = FeatureStatistics(train_data_path, threshold, config)
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
        empirical_counts = csr_matrix(np.zeros(self.feature_statistics.num_features, dtype=np.float64))
        feature_vectors = [self.feature_statistics.hist_to_feature_vec_dict[hist] for sentence in
                           self.feature_statistics.history_sentence_list for hist in sentence]

        for feature_vector in feature_vectors:
            empirical_counts += feature_vector

        return empirical_counts

    @staticmethod
    def calc_normalization_term(v, sentence_history_list, hist_to_all_tag_feature_matrix_dict):
        norm = 0.
        for sentence in sentence_history_list:
            for hist in sentence:
                new_hist_key = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag, ctag=None, nword=hist.nword,
                                       pword=hist.pword, nnword=hist.nnword, ppword=hist.ppword)
                mat = hist_to_all_tag_feature_matrix_dict[new_hist_key]
                norm += np.log(np.sum(np.exp(mat @ v)))
        return norm

    @staticmethod
    def calc_expected_counts(history_sentence_list, v, hist_to_all_tag_feature_matrix_dict):
        expected_counts = np.zeros(v.shape, np.float64)
        for sentence in history_sentence_list:
            for hist in sentence:
                new_hist_key = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag, ctag=None, nword=hist.nword,
                                       pword=hist.pword, nnword=hist.nnword, ppword=hist.ppword)
                mat = hist_to_all_tag_feature_matrix_dict[new_hist_key]
                exp_mat = np.exp(mat @ v)
                prob_mat = exp_mat / np.sum(exp_mat)
                # expected_counts += prob_mat * mat
                # expected_counts += np.squeeze(np.asarray(np.sum(mat.multiply(prob_mat[:, None]), axis=0)))
                expected_counts += mat.T @ (prob_mat)

        return expected_counts

    @staticmethod
    def calc_objective_per_iter(*args):
        starttime = timeit.default_timer()
        v = args[0]
        reg_lambda = args[2]
        empirical_counts = args[3]
        sentence_history_list = args[4]
        hist_to_all_tag_feature_matrix_dict = args[7]
        linear_term = empirical_counts.dot(v).item()
        normalization_term = MaximumEntropyMarkovModel.calc_normalization_term(
            v, sentence_history_list, hist_to_all_tag_feature_matrix_dict
        )

        regularization_term = 0.5 * reg_lambda * np.linalg.norm(v, ord=2)
        likelihood = linear_term - normalization_term - regularization_term

        # FINISHED CALCULATING LIKELIHOOD
        expected_counts = MaximumEntropyMarkovModel.calc_expected_counts(
            sentence_history_list, v, hist_to_all_tag_feature_matrix_dict
        )
        regularization_grad = reg_lambda * v
        grad = empirical_counts - expected_counts - regularization_grad

        # FINISHED CALCULATING GRAD
        stoptime = timeit.default_timer()
        print(f'execution time is : {stoptime - starttime}')
        return (-1) * likelihood, (-1) * grad

    def optimize_model(self):
        arg_1 = (self.feature_statistics.hist_to_feature_vec_dict)
        arg_2 = self.reg_lambda
        arg_3 = self.calc_empirical_counts()
        arg_4 = self.feature_statistics.history_sentence_list
        arg_5 = self.feature_statistics.word_possible_tag_set
        arg_6 = self.feature_statistics.word_possible_tag_with_threshold_dict
        arg_7 = self.feature_statistics.hist_to_all_tag_feature_matrix_dict
        args = (arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)
        # w_0 = np.random.normal(0, 0.01, (self.feature_statistics.num_features)).astype(np.float64)
        w_0 = np.zeros(self.feature_statistics.num_features, dtype=np.float64)
        optimal_params = fmin_l_bfgs_b(func=self.calc_objective_per_iter, x0=w_0, args=args, maxiter=10000, iprint=1)
        weights = optimal_params[0]
        print(weights)
        weights_dir = os.path.join(self.dump_weights_path, str(self.feature_statistics.threshold) +
                                   f'-reg_lambda={arg_2}')
        self.v = weights
        with open(weights_dir, 'wb') as f:
            pickle.dump(weights, f)