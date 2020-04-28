import numpy as np
import pickle
import os
from features import History
from utils import timeit


class Viterbi:
    def __init__(self, v, sentence_hist_list: [[History]], tags_set, all_possible_tags_dict,
                 get_feature_from_hist, word_possible_tag_set, word_possible_tag_with_threshold_dict, prob_dict,
                 exp_dict, threshold, reg_lambda):
        self.v = v
        self.sentence_list = sentence_hist_list
        tags_set.add('*')
        self.tags_set = tags_set
        self.all_possible_tags_dict = all_possible_tags_dict
        self.get_feature_from_hist = get_feature_from_hist
        self.word_possible_tag_set = word_possible_tag_set
        self.word_possible_tag_with_threshold_dict = word_possible_tag_with_threshold_dict
        tag_list = list(tags_set)
        self.tag_to_index = {tag_list[i]: i for i in range(len(tag_list))}
        self.index_to_tag = {v: k for k, v in self.tag_to_index.items()}
        self.pi_tables = None
        self.bp_tables = None
        self.prob_dict = prob_dict
        self.exp_dict = exp_dict
        self.res_path = 'res'
        self.dump_name = 'test1'
        self.threshold = threshold
        self.reg_lambda = reg_lambda
        self.known_words = {word for word in self.word_possible_tag_set}
        self.unknown_words = set()

    def predict_all_test(self):
        print('starting inference')
        all_res_tags = []
        all_acc_list = []
        all_tagged_res_list = []# will be saved to file
        all_gt_tags = []
        all_right_tag_list=[]
        for num, sentence in enumerate(self.sentence_list):
            if num + 1 % 10 == 0:
                print(f'handling sentence number {num+1}')
            cur_res = self.predict(sentence)
            all_res_tags.append(cur_res)
            ground_truth = [hist.ctag for hist in sentence]
            all_gt_tags.append(ground_truth)
            res_acc, right_tag_list = self.calc_accuracy(cur_res, ground_truth)
            print(f'accuracy for sentence {num+1}: {res_acc}')
            all_acc_list.append(res_acc)
            all_right_tag_list += right_tag_list
            cur_tagged_res = []
            for hist, tag in zip(sentence, cur_res):
                cword = hist.cword
                cur_tagged_res.append(cword + '_' + tag)
            all_tagged_res_list.append(cur_tagged_res)

            assert len(cur_res) == len(sentence)
        # print(all_tagged_res_list)
        # print(f'total accuracy: {sum(all_acc_list)/len(all_acc_list)}')
        print(f'total accuracy: {sum(all_right_tag_list) / len(all_right_tag_list) * 100}')
        self.dump_res(all_tagged_res_list, all_gt_tags, all_res_tags)
        return all_res_tags, all_acc_list

    @staticmethod
    def calc_accuracy(res, gt):
        right_tag_list = [1 if res_tag == true_tag else 0 for res_tag, true_tag in zip(res, gt)]
        acc = (sum(right_tag_list)/len(right_tag_list)) * 100
        return acc, right_tag_list

    def get_possible_tag_set_from_word(self, word):
        w_lower = word.lower()
        w_first_upper = list(word)
        w_first_upper[0] = w_first_upper[0].upper()
        w_first_upper = ''.join(w_first_upper)

        if self.word_possible_tag_with_threshold_dict.get(word, None):
            tag_set = {self.word_possible_tag_with_threshold_dict[word][0]}
        elif self.word_possible_tag_with_threshold_dict.get(w_lower, None):
            tag_set = {self.word_possible_tag_with_threshold_dict[w_lower][0]}
        elif self.word_possible_tag_with_threshold_dict.get(w_first_upper, None):
            tag_set = {self.word_possible_tag_with_threshold_dict[w_first_upper][0]}

        elif self.word_possible_tag_set.get(word, None):
            tag_set = self.word_possible_tag_set[word]
        elif self.word_possible_tag_set.get(w_lower, None):
            tag_set = self.word_possible_tag_set[w_lower]
        elif self.word_possible_tag_set.get(w_first_upper, None):
            tag_set = self.word_possible_tag_set[w_first_upper]
        else:  # this is a new word
            tag_set = self.tags_set - {'*'}
        return tag_set

    @timeit
    def fill_prob_dict_from_sentence(self, sentence):
        for idx, hist in enumerate(sentence):
            if idx == 0:
                pptag_set = {'*'}
                ptag_set = {'*'}

            ctag_set = self.get_possible_tag_set_from_word(hist.cword)
            for pptag in pptag_set:
                for ptag in ptag_set:
                    norm_i = 0.
                    cur_possible_hist_list = []
                    for c_tag in ctag_set:
                        n_hist = History(cword=hist.cword, pptag=pptag, ptag=ptag,
                                         nword=hist.nword, pword=hist.pword, ctag=c_tag)
                        if not self.prob_dict.get(n_hist, None):
                            if not self.all_possible_tags_dict.get(n_hist, None):
                                self.all_possible_tags_dict[n_hist] = self.get_feature_from_hist(n_hist)
                            dot_prod = np.sum(self.v[self.all_possible_tags_dict[n_hist]])
                            self.exp_dict[n_hist] = np.exp(dot_prod).astype(np.float128)

                        cur_possible_hist_list.append(n_hist)
                        norm_i += self.exp_dict[n_hist]

                    # fill prob_dict
                    for hist in cur_possible_hist_list:
                        if not self.prob_dict.get(hist, None):
                            if len(cur_possible_hist_list) == 1:
                                self.prob_dict[hist] = 1
                            else:
                                try:
                                    self.prob_dict[hist] = np.float128(self.exp_dict[hist] / norm_i)
                                except:
                                    print(f'cword: {hist.cword}')
                                    print(f'cur tag set: {ctag_set}')
                                    print(f'self.exp_dict[hist]: {self.exp_dict[hist]}')
                                    print(f'norm_i: {norm_i}')
                                    raise Exception
            pptag_set = ptag_set
            ptag_set = ctag_set

    def calc_res_tags(self, sentence):
        # print('calculating pi')
        for ind, k in enumerate(range(1, len(sentence) + 1)):
            cur_hist = sentence[ind]
            if ind == 0:
                pp_tag_set = {'*'}
                p_tag_set = {'*'}
            cur_tag_set = self.get_possible_tag_set_from_word(cur_hist.cword)

            for v in cur_tag_set:
                for u in p_tag_set:
                    max_pi_mul_q_val = -np.inf
                    max_t_index = 0
                    for t in pp_tag_set:
                        t_index = self.tag_to_index[t]
                        new_hist = History(cword=cur_hist.cword, pptag=t, ptag=u,
                                           ctag=v, nword=cur_hist.nword, pword=cur_hist.pword)
                        if np.isclose(self.prob_dict[new_hist], 0.):
                            q = -np.inf
                        else:
                            q = np.log(self.prob_dict[new_hist])
                        pi = self.pi_tables[k - 1, t_index, self.tag_to_index[u]]
                        res = q + pi
                        if res > max_pi_mul_q_val:
                            max_pi_mul_q_val = res
                            max_t_index = t_index

                    self.pi_tables[k, self.tag_to_index[u], self.tag_to_index[v]] = max_pi_mul_q_val
                    self.bp_tables[k, self.tag_to_index[u], self.tag_to_index[v]] = max_t_index
            pp_tag_set = p_tag_set
            p_tag_set = cur_tag_set
        max_ind = np.argmax(self.pi_tables[-1, :, :])

        t_n_m_1, t_n = np.unravel_index(max_ind, self.pi_tables[-1, :, :].shape)
        res_numbers = [t_n, t_n_m_1]
        # print('calculating bp')
        for ind, k in enumerate(reversed(range(1, len(sentence) - 1))):
            append_idx = self.bp_tables[k + 2, res_numbers[ind + 1], res_numbers[ind]]
            res_numbers.append(append_idx)

        res_tags = list(reversed([self.index_to_tag[res] for res in res_numbers]))
        return res_tags

    def predict(self, sentence):
        self.pi_tables = np.full(shape=(len(sentence) + 1, len(self.tags_set), len(self.tags_set)), fill_value=-np.inf)
        self.pi_tables[0, self.tag_to_index["*"], self.tag_to_index["*"]] = 0.
        self.bp_tables = np.full(shape=self.pi_tables.shape, fill_value=10**2, dtype=np.int)
        self.fill_prob_dict_from_sentence(sentence)
        res_tags = self.calc_res_tags(sentence)
        return res_tags

    def dump_res(self, all_tagged_res_list, all_gt_tags, all_res_tags):
        if not os.path.isdir(self.res_path):
            os.makedirs(self.res_path)
        dump_name = self.dump_name + '_' + str(self.threshold) + '_' + str(self.reg_lambda)
        dump_path = os.path.join(self.res_path, dump_name)
        with open(dump_path, 'wb') as f:
            res = (all_tagged_res_list, all_gt_tags, all_res_tags)
            pickle.dump(res, f)
