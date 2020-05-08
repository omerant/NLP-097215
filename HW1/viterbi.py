import numpy as np
import pickle
import os
from utils import History
from utils import timeit, MIN_EXP_VAL, MIN_LOG_VAL, BASE_PROB, OpenClassTypes, UNKNOWN_WORD
np.random.seed(0)


class Viterbi:
    def __init__(self, v, sentence_hist_list: [[History]], tags_list, all_possible_tags_dict,
                 get_feature_from_hist, word_possible_tag_set, word_possible_tag_with_threshold_dict,
                 rare_words_tags, threshold, reg_lambda, beam_width=6):
        self.v = v
        self.sentence_list = sentence_hist_list
        tags_list.append('*')
        self.tags_list = tags_list
        self.all_possible_tags_dict = all_possible_tags_dict
        self.get_feature_from_hist = get_feature_from_hist
        self.word_possible_tag_set = word_possible_tag_set
        self.word_possible_tag_with_threshold_dict = word_possible_tag_with_threshold_dict
        self.rare_words_tags = rare_words_tags
        self.tag_to_index = {self.tags_list[i]: i for i in range(len(self.tags_list))}
        self.index_to_tag = {v: k for k, v in self.tag_to_index.items()}
        self.pi_tables = None
        self.bp_tables = None
        self.prob_dict = dict()
        self.res_path = 'res'
        self.dump_name = 'test1'
        self.threshold = threshold
        self.reg_lambda = reg_lambda
        self.beam_width = beam_width

    def predict_all_test(self):
        print('starting inference')
        all_res_tags = []
        all_acc_list = []
        all_tagged_res_list = []# will be saved to file
        all_gt_tags = []
        all_gt_tags_known = []
        all_gt_tags_unknown = []

        all_right_tag_list = []
        all_right_tag_list_known = []
        all_right_tag_list_unknown = []

        for num, sentence in enumerate(self.sentence_list):
            if num + 1 % 10 == 0:
                print(f'handling sentence number {num+1}')
            cur_res = self.predict(sentence)
            cur_res_known = [tag_res for hist, tag_res in zip(sentence, cur_res) if self.word_possible_tag_set.get(hist.cword, None)]
            cur_res_unknown = [tag_res for hist, tag_res in zip(sentence, cur_res) if not self.word_possible_tag_set.get(hist.cword, None)]
            all_res_tags.append(cur_res)

            ground_truth = [hist.ctag for hist in sentence]
            ground_truth_known = [hist.ctag for hist in sentence if self.word_possible_tag_set.get(hist.cword, None)]
            ground_truth_unknown = [hist.ctag for hist in sentence if not self.word_possible_tag_set.get(hist.cword, None)]
            all_gt_tags.append(ground_truth)
            all_gt_tags_known.append(ground_truth_known)
            all_gt_tags_unknown.append(ground_truth_unknown)
            res_acc, right_tag_list = self.calc_accuracy(cur_res, ground_truth)
            res_acc_known, right_tag_list_known = self.calc_accuracy(cur_res_known, ground_truth_known)
            res_acc_unknown, right_tag_list_unknown = self.calc_accuracy(cur_res_unknown, ground_truth_unknown)

            print(f'accuracy for sentence {num+1}: {res_acc}')
            print(f'known words accuracy for sentence {num + 1}: {res_acc_known}')
            print(f'unknown words accuracy for sentence {num + 1}: {res_acc_unknown}')
            all_acc_list.append(res_acc)
            all_right_tag_list += right_tag_list
            all_right_tag_list_known += right_tag_list_known
            all_right_tag_list_unknown += right_tag_list_unknown
            cur_tagged_res = []
            for hist, tag in zip(sentence, cur_res):
                cword = hist.cword
                cur_tagged_res.append(cword + '_' + tag)
            all_tagged_res_list.append(cur_tagged_res)

            assert (len(cur_res) == len(sentence)), f'cur res: {cur_res}, sentence: {sentence}'
        # print(all_tagged_res_list)
        # print(f'total accuracy: {sum(all_acc_list)/len(all_acc_list)}')
        print(f'precent of known words in corpus: {100 * len(all_right_tag_list_known)/len(all_right_tag_list)}')
        print(f'precent of unknown words in corpus: {100 * len(all_right_tag_list_unknown) / len(all_right_tag_list)}')
        print(f'total accuracy: {self._calc_acc(all_right_tag_list)}')
        print(f'known words accuracy: {self._calc_acc(all_right_tag_list_known)}')
        print(f'unknown words accuracy: {self._calc_acc(all_right_tag_list_unknown)}')
        self.dump_res(all_tagged_res_list, all_gt_tags, all_res_tags, self.sentence_list)
        return all_res_tags, all_acc_list

    @staticmethod
    def _calc_acc(binary_list):
        if sum(binary_list) == 0:
            return 0.
        return sum(binary_list) / len(binary_list) * 100

    @staticmethod
    def calc_accuracy(res, gt):
        right_tag_list = [1 if res_tag == true_tag else 0 for res_tag, true_tag in zip(res, gt)]
        if len(right_tag_list) > 0:
            acc = (sum(right_tag_list)/len(right_tag_list)) * 100
        else:
            acc = 0.
        return acc, right_tag_list

    def get_possible_tag_set_from_word(self, word):
        tag_set = None
        if self.word_possible_tag_with_threshold_dict.get(word, None):
            tag_set = {self.word_possible_tag_with_threshold_dict[word][0]}
        else:
            tag_set = self.tags_list[:-1]
        return tag_set

    @timeit
    def fill_prob_dict_from_sentence(self, sentence):
        for idx, hist in enumerate(sentence):
            if idx == 0:
                pptag_set = {'*'}
                ptag_set = {'*'}

            ctag_set = self.get_possible_tag_set_from_word(hist.cword)

            filtered_ctag_set = None
            cur_hist_list = []
            for c_tag in ctag_set:
                for ptag in ptag_set:
                    for pptag in pptag_set:
                        n_hist = History(cword=hist.cword, pptag=pptag, ptag=ptag,
                                         nword=hist.nword, pword=hist.pword, ctag=c_tag,
                                         nnword=hist.nnword, ppword=hist.ppword)

                        dot_prod = np.sum(self.v[self.get_feature_from_hist(n_hist)])

                        cur_hist_list.append((n_hist, dot_prod))
            slice_idx = min(self.beam_width, len(ctag_set))
            sorted_possible_hist_list = list(sorted(cur_hist_list, key=lambda x: x[1], reverse=True))[:slice_idx]
            filtered_ctag_set = {h.ctag for h, _ in sorted_possible_hist_list}
            filtered_hist_list = [h for h, _ in cur_hist_list]
            dot_p_arr = np.array([dot_p for _, dot_p in cur_hist_list]).astype(np.float64)
            try:
                exp_arr = np.exp(dot_p_arr-np.max(dot_p_arr)).astype(np.float64)
            # fill prob_dict
                prob_arr = exp_arr/np.sum(exp_arr)
                for hist, prob in zip(filtered_hist_list, prob_arr):
                    self.prob_dict[hist] = prob

                pptag_set = ptag_set
                ptag_set = filtered_ctag_set
            except FloatingPointError:
                print(f'EXCEPTION FloatingPointError WAS RAISED')
                max_hist = filtered_hist_list[np.argmax(dot_p_arr)]
                self.prob_dict[max_hist] = 1

    def calc_res_tags(self, sentence):
        # print('calculating pi')
        for ind, k in enumerate(range(1, len(sentence) + 1)):
            cur_hist = sentence[ind]

            if ind == 0:
                pp_tag_set = {'*'}
                p_tag_set = {'*'}

            cur_tag_set = self.get_possible_tag_set_from_word(cur_hist.cword)

            for v in cur_tag_set:
                if v == 13 or v == '13':
                    print(f'cur tag set: {cur_tag_set}')
                for u in p_tag_set:
                    max_pi_mul_q_val = -np.inf
                    max_t_index = self.tag_to_index['NN']#10**3
                    for t in pp_tag_set:
                        t_index = self.tag_to_index[t]
                        new_hist = History(cword=cur_hist.cword, pptag=t, ptag=u,
                                           ctag=v, nword=cur_hist.nword, pword=cur_hist.pword,
                                           nnword=cur_hist.nnword, ppword=cur_hist.ppword)
                        # if self.prob_dict[new_hist] < MIN_LOG_VAL:
                        #     q = -np.inf
                        # else:
                        #     q = np.log(self.prob_dict[new_hist])
                        if not self.prob_dict.get(new_hist, None):
                            q = -np.inf
                        else:
                            q = np.log(self.prob_dict[new_hist])

                        pi = self.pi_tables[k - 1, t_index, self.tag_to_index[u]]
                        res = q + pi
                        if res > max_pi_mul_q_val:
                            max_pi_mul_q_val = res
                            max_t_index = t_index
                    if max_t_index == 10**2:
                        print(f'CURRENT WORD: {cur_hist.cword}')
                        raise Exception()
                    # assert max_t_index != 10**3
                    # print(f'max index: {max_t_index}')
                    try:
                        self.pi_tables[k, self.tag_to_index[u], self.tag_to_index[v]] = max_pi_mul_q_val
                        self.bp_tables[k, self.tag_to_index[u], self.tag_to_index[v]] = max_t_index
                    except:
                        print(f' k is {k}')
                        print(f' u is {u}, index is {self.tag_to_index[u]}')
                        print(f' v is {v}, ')
                        print(f' current word is {cur_hist.cword}')
                        raise Exception('borat')
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

    @timeit
    def predict(self, sentence):
        self.pi_tables = np.full(shape=(len(sentence) + 1, len(self.tags_list), len(self.tags_list)), fill_value=-np.inf)
        self.pi_tables[0, self.tag_to_index["*"], self.tag_to_index["*"]] = 0.
        self.bp_tables = np.full(shape=self.pi_tables.shape, fill_value=10**2, dtype=np.int)
        self.fill_prob_dict_from_sentence(sentence)
        res_tags = self.calc_res_tags(sentence)
        return res_tags

    def dump_res(self, all_tagged_res_list, all_gt_tags, all_res_tags, sentence_list):
        if not os.path.isdir(self.res_path):
            os.makedirs(self.res_path)
        dump_name = self.dump_name + '_' + str(self.threshold) + '_' + str(self.reg_lambda)
        dump_path = os.path.join(self.res_path, dump_name)
        with open(dump_path, 'wb') as f:
            res = (all_tagged_res_list, all_gt_tags, all_res_tags, sentence_list)
            pickle.dump(res, f)
