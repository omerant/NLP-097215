import numpy as np
import pickle
import os
from features import History

#genrally - t is TAGi-2, u is TAGi-1, v is currentTAG
#probability func
#common_features func
#all_tags starts from 0 with '*'


class Viterbi:
    def __init__(self, v, sentence_hist_list: [[History]], tags_set, all_possible_tags_dict, get_feature_from_hist):
        self.v = v
        self.sentence_list = sentence_hist_list
        self.tags_set = tags_set
        self.all_possible_tags_dict = all_possible_tags_dict
        self.get_feature_from_hist = get_feature_from_hist
        tag_list = list(tags_set)
        self.tag_to_index = {tag_list[i]: i for i in range(len(tag_list))}
        self.index_to_tag = {v: k for k, v in self.tag_to_index.items()}
        self.pi_tables = None
        self.bp_tables = None
        self.prob_dict = dict()
        self.exp_dict = dict()

    def predict_all(self):
        print('starting inference')
        all_res_tags = []
        all_acc_list = []
        all_tagged_res_list = []# will be saved to file
        all_gt_tags = []
        for num, sentence in enumerate(self.sentence_list):
            cur_res = self.predict(sentence)
            all_res_tags.append(cur_res)
            ground_truth = [hist.ctag for hist in sentence]
            all_gt_tags.append(ground_truth)

            res_acc = self.calc_accuracy(cur_res, ground_truth)
            print(f'accuracy for sentence {num+1}: {res_acc}')
            all_acc_list.append(res_acc)
            cur_tagged_res = []
            for hist, tag in zip(sentence, cur_res):
                cword = hist.cword
                cur_tagged_res.append(cword + '_' + tag)
            all_tagged_res_list.append(cur_tagged_res)
            assert len(cur_res) == len(sentence)
        # print(all_tagged_res_list)
        print(f'total accuracy: {sum(all_acc_list)/len(all_acc_list)}')
        res_handler = ResultsHandler(all_tagged_res_list, all_gt_tags, all_res_tags)
        res_handler.dump_res()
        return all_res_tags, all_acc_list

    @staticmethod
    def calc_accuracy(res, gt):
        right_tag_list = [1 if res_tag == true_tag else 0 for res_tag, true_tag in zip(res, gt)]
        acc = (sum(right_tag_list)/len(right_tag_list)) * 100
        return acc

    def predict(self, sentence):
        self.pi_tables = np.full(shape=(len(sentence) + 1, len(self.tags_set), len(self.tags_set)), fill_value=0.)
        self.pi_tables[0, self.tag_to_index["*"], self.tag_to_index["*"]] = 1
        self.bp_tables = np.full(shape=self.pi_tables.shape, fill_value=10**20)

        # print('calculating prob_dict')

        for hist in sentence:
            # print(f'cword = {hist.cword}')
            norm_i = 0.
            cur_possible_hist_list = []
            # fill exp dict
            for pptag in self.tags_set:
                for ptag in self.tags_set:
                    for c_tag in self.tags_set:
                        n_hist = History(cword=hist.cword, pptag=pptag, ptag=ptag,
                                         nword=hist.nword, pword=hist.pword, ctag=c_tag)
                        if not self.exp_dict.get(n_hist, None):
                            if not self.all_possible_tags_dict.get(n_hist, None):
                                self.all_possible_tags_dict[n_hist] = self.get_feature_from_hist(n_hist)
                            dot_prod = np.sum(self.v[self.all_possible_tags_dict[n_hist]])
                            self.exp_dict[n_hist] = np.exp(dot_prod)
                            cur_possible_hist_list.append(n_hist)

                        norm_i += self.exp_dict[n_hist]
                    # fill prob_dict
                    for hist in cur_possible_hist_list:
                        self.prob_dict[hist] = self.exp_dict[hist] / norm_i

        # print('calculating pi and bp')

        for ind, k in enumerate(range(1, len(sentence) + 1)):

            # print(f'k: {k}, curr_hist: {cur_hist}')
            cur_hist = sentence[ind]
            cur_tag_set = self.tags_set - {'*'}
            p_tag_set = self.tags_set - {'*'} if k > 1 else {'*'}
            pp_tag_set = self.tags_set - {'*'} if k > 2 else {'*'}
            for v in cur_tag_set:
                for u in p_tag_set:
                    max_pi_mul_q_val = 0.
                    max_t_index = 0
                    for t in pp_tag_set:
                        t_index = self.tag_to_index[t]
                        new_hist = History(cword=cur_hist.cword, pptag=t, ptag=u,
                                           ctag=v, nword=cur_hist.nword, pword=cur_hist.pword)
                        q = self.prob_dict[new_hist]
                        pi = self.pi_tables[k - 1, t_index, self.tag_to_index[u]]
                        res = q * pi
                        if res > max_pi_mul_q_val:
                            max_pi_mul_q_val = res
                            max_t_index = t_index

                    self.pi_tables[k, self.tag_to_index[u], self.tag_to_index[v]] = max_pi_mul_q_val
                    self.bp_tables[k, self.tag_to_index[u], self.tag_to_index[v]] = max_t_index

        max_ind = np.argmax(self.pi_tables[-1, :, :])

        t_n_m_1, t_n = np.unravel_index(max_ind, self.pi_tables[-1, :, :].shape)
        res_numbers = [t_n, t_n_m_1]

        for ind, k in enumerate(reversed(range(1, len(sentence) - 1))):
            # print(f'ind: {ind}, k: {k}')
            append_idx = self.bp_tables[k + 2, res_numbers[ind], res_numbers[ind + 1]]
            # print(f'append index: {append_idx}')
            # print(f'index in bp table: {k + 2}')
            res_numbers.append(append_idx)

        res_tags = list(reversed([self.index_to_tag[res] for res in res_numbers]))

        # print(res_tags)

        return res_tags


class ResultsHandler:
    def __init__(self, all_tagged_res_list=None, all_gt_tags=None, all_res_tags=None):
        self.res_path = 'res'
        self.dump_name = 'test1'
        self.all_tagged_res_list = all_tagged_res_list
        self.all_gt_tags = all_gt_tags
        self.all_res_tags = all_res_tags

    def dump_res(self):
        if not os.path.isdir(self.res_path):
            os.makedirs(self.res_path)
        dump_path = os.path.join(self.res_path, self.dump_name)
        with open(dump_path, 'wb') as f:
            res = (self.all_tagged_res_list, self.all_gt_tags, self.all_res_tags)
            pickle.dump(res, f)

    def get_res(self):
        dump_path = os.path.join(self.res_path, self.dump_name)
        with open(dump_path, 'rb') as f:
            self.all_tagged_res_list, self.all_gt_tags, self.all_res_tags = pickle.load(f)

# res_handler = ResultsHandler()
# res_handler.get_res()
# pass