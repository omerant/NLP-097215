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
        self.prob_func = dict()
        # self.v = np.full(self.feature_statistics.num_features, 0., dtype=np.float64)
        # self.v[1] = 0.01# start with uniform distribution
        # self.prob_func = self.load_prob_dict()
        self.v = self._load_v_from_pickle()

    def _load_v_from_pickle(self):
        print('filling probability dict')
        weights_dir = os.path.join(self.dump_weights_path, str(self.feature_statistics.version) +
                                   f'-reg_lambda={1}-num_iter={1}')
        with open(weights_dir, 'rb') as f:
            res = pickle.load(f).reshape(-1)

        return res

    def fill_prob_dict(self):
        print('filling probability dict')
        v = self.v / np.linalg.norm(self.v, ord=2)
        for hist in self.feature_statistics.all_possible_tags_dict.keys():
            res = np.exp(np.sum(v[self.feature_statistics.all_possible_tags_dict[hist]]))
            sum_all_tag_res = 0.
            for ctag in self.feature_statistics.tags_set:
                new_hist = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag,
                                   ctag=ctag, nword=hist.nword, pword=hist.pword)
                sum_all_tag_res += np.exp(np.sum(v[self.feature_statistics.all_possible_tags_dict[new_hist]]))

            self.prob_func[hist] = res / sum_all_tag_res
        dump_path = os.path.join('probability_dict', f'version={self.feature_statistics.version}_threshold={self.feature_statistics.threshold}')
        with open(dump_path, 'wb') as f:
            pickle.dump(self.prob_func, f)

        print('finished filling probability dict')

    def load_prob_dict(self):
        print('loading probability dict')
        dump_path = os.path.join('probability_dict', f'version={self.feature_statistics.version}_threshold={self.feature_statistics.threshold}')
        with open(dump_path, 'rb') as f:
            self.prob_func = pickle.load(f)

        print('finished loading probability dict')

    def calc_empirical_counts(self):
        print('----calculating empirical counts-----')
        features_linear_term_indices = [self.feature_statistics.all_possible_tags_dict[hist] for hist in
                                        self.feature_statistics.history_ordered_list]
        print(f'----finished getting features len_all_feature={len(features_linear_term_indices)}-----')
        empirical_counts = np.full(self.v.shape, 0)
        for feature_ind in features_linear_term_indices:
            empirical_counts[feature_ind] += 1

        dump_path = os.path.join('empirical_counts', f'version={self.feature_statistics.version}_threshold={self.feature_statistics.threshold}')
        with open(dump_path, 'wb') as f:
            pickle.dump(empirical_counts, f)
        print('----finished calculating empirical counts-----')
        return empirical_counts

    def calc_likelihood_and_grad_one_iteration(*args):
        # print(args)
        model = args[0]
        model.v = args[1]
        v = args[1]
        # v_norm = np.linalg.norm(v, ord=2)
        v = v
        print(f'v: {v}')
        reg_lambda = args[3]
        num_iter = args[5]
        history_batch = model.feature_statistics.history_ordered_list
        tags_set = model.feature_statistics.tags_set

        # print('-----calculating linear term-----')
        empirical_counts = args[4]
        linear_term = empirical_counts.dot(v)
        # print(linear_term)
        # print('-----finished calculating linear term-----')
        # #
        # # calc Normalization term
        # print('-----calculating normalization term-----')

        def calc_normalization_term(sampled_hist):
            norm_term = 0
            prob_denominator_ordered_list = []
            exp_dot_dict = dict()  # dict of the form: {history(x_i, y): exp(v*f(x_i, y'))}

            for hist in sampled_hist:
                norm_i = []
                for ctag in tags_set:
                    new_hist = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag,
                                       ctag=ctag, nword=hist.nword, pword=hist.pword)
                    feature_vec_indices_new_hist = model.feature_statistics.all_possible_tags_dict[new_hist]
                    dot_product = np.sum(v[feature_vec_indices_new_hist])
                    exp_sum_res = np.exp(dot_product)
                    exp_dot_dict[new_hist] = exp_sum_res
                    norm_i.append(exp_sum_res)

                inner_sum = np.sum(norm_i)
                prob_denominator_ordered_list.append(inner_sum)
                norm_term += np.log(inner_sum)
            return norm_term, prob_denominator_ordered_list, exp_dot_dict

        # v_norm = np.linalg.norm(v, ord=2)
        normalization_term, prob_denominator_ordered_list, exp_dict = calc_normalization_term(
            sampled_hist=history_batch
        )
        # print(normalization_term)
        # print('-----finished calculating normalization term-----')
        #
        # print('-----calculating regularization_term term-----')
        regularization = 0.5 * reg_lambda * np.linalg.norm(v, ord=2)
        # print(regularization)
        # print('-----finished calculating regularization_term-----')

        likelihood = (1 / model.feature_statistics.num_sentences) * (linear_term - normalization_term - regularization)
        # print(likelihood)
        # print('-----calculating empirical counts-----')
        empirical_counts = args[4]
        empirical_counts = empirical_counts
        # print(empirical_counts)
        # print('-----finished calculating empirical counts-----')

        def calc_expected_counts(prob_denom_list, exp_d):
            expected_counts = np.full(v.shape, 0.)
            for hist, prob_denominator in zip(history_batch, prob_denom_list):
                for ctag in tags_set:

                    new_hist = History(cword=hist.cword, pptag=hist.pptag, ptag=hist.ptag,
                                       ctag=ctag, nword=hist.nword, pword=hist.pword)
                    feature_vec_indices_new_hist = model.feature_statistics.all_possible_tags_dict[new_hist]
                    exp_res = exp_d[new_hist]
                    prob = (exp_res / prob_denominator)
                    expected_counts[feature_vec_indices_new_hist] += 1
                    expected_counts[feature_vec_indices_new_hist] *= prob

            return np.array(expected_counts)

        # print('-----calculating expected counts-----')
        expected_counts = calc_expected_counts(prob_denominator_ordered_list, exp_dict)
        # print(expected_counts)
        # print('-----finished calculating expected counts-----')
        # calc regularization_grad
        # print('-----calculating regularization_grad-----')
        regularization_grad = reg_lambda * v
        # print(regularization_grad)
        # print('-----finished calculating regularization_grad-----')

        grad = (1 / model.feature_statistics.num_sentences) * (empirical_counts - expected_counts - regularization_grad)
        print(f'grad: {grad}')
        return (-1) * likelihood, (-1) * grad


class Optimizer:

    def __init__(self, num_models, train_data_path, num_iter=1):
        self.num_models = num_models
        self.model = MaximumEntropyMarkovModel(train_data_path, test_data_path=None)
        self.num_iter = num_iter
        self.args = self.get_args()

    def get_args(self):
        # each param list is of form [reg_lambda]
        empirical_counts = self.model.calc_empirical_counts()
        lambda_reg = 1
        args = [[self.model, lambda_reg, empirical_counts]]
        return args

    @staticmethod
    def optimize_model(args):
        cur_args = args

        print(args)
        model = cur_args[0]
        print(f'V: {model.v}')
        reg_lambda = cur_args[1]
        cur_iter = cur_args[3]
        print(f'starting oprtimization with lambda = {reg_lambda}')
        # bounds = [(-np.inf, 1) for i in range(model.feature_statistics.num_features)]
        optimal_params = fmin_l_bfgs_b(func=model.calc_likelihood_and_grad_one_iteration, x0=model.v, args=cur_args,
                                       iprint=1)
        model.v = optimal_params[0]
        print(f'model params: {model.v}')
        weights_dir = os.path.join(model.dump_weights_path, str(model.feature_statistics.version) +
                                   f'-reg_lambda={reg_lambda}-num_iter={cur_iter}')
        with open(weights_dir, 'wb') as f:
            pickle.dump(model.v, f)

    def optimize_num_iter(self):
        print(f'Strarting optimization ')
        self.update_iteration_num_in_args(1)
        self.optimize_model(self.args[0])

    def update_iteration_num_in_args(self, it_num):
        self.args = [[args[0], args[1], args[2], it_num] for args in self.args]


if __name__ == '__main__':
    train1_path = 'data/train1_short.wtag'
    test1_short_path = 'data/train1_short.wtag'
    import timeit
    start = timeit.default_timer()
    optimizer = Optimizer(num_models=1, train_data_path=train1_path, num_iter=1)
    optimizer.optimize_num_iter()
    memm = MaximumEntropyMarkovModel(train_data_path=train1_path, test_data_path=test1_short_path)
    memm.load_prob_dict()

    vit = Viterbi(memm.prob_func,
                  sentence_hist_list=memm.test_data.history_sentence_list,
                  tags_set=memm.feature_statistics.tags_set
    )
    all_res_list, all_acc_list = vit.predict_all()
    print(all_res_list)
    print(all_acc_list)
    print(f'avg acc: {sum(all_acc_list)/len(all_acc_list)}')

    stop = timeit.default_timer()
    print(f'execution time for one iteration: {stop - start}')
