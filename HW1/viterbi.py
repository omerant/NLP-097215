import numpy as np
import pickle
import os
import multiprocessing as mp
from utils import History
np.random.seed(0)


class Viterbi:
    def __init__(self, v, sentence_hist_list: [[History]], tags_list, all_possible_tags_dict,
                 get_feature_from_hist, word_possible_tag_set, word_possible_tag_with_threshold_dict,
                 rare_words_tags, threshold, reg_lambda, beam_width=6):
        self.v = v
        self.sentence_list = sentence_hist_list
        tags_list.append('*')
        tags_list.append('**')
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

    def _prep_args(self, num_workers, q):
        step_size = int(len(self.sentence_list) / num_workers)
        # each argument tuple is (i, hist_sentence_list, q, self)
        args_list = [(i, self.sentence_list[i * step_size: (i + 1) * step_size], q, self) for i in
                     range(num_workers)]
        return args_list

    @staticmethod
    def _calc_predict(i, sentence_list, q: mp.Queue, self):
        """
        method that executes viterbi on given sentence list
        :param i: index that will be used to sort sentences after process finished
        :param sentence_list: list of sentences to work on
        :param q: queue in which we will save results
        :param self: viterbi object
        :return:
        """
        all_res_tags = []
        all_acc_list = []
        all_tagged_res_list = []# will be saved to file
        all_gt_tags = []
        all_gt_tags_known = []
        all_gt_tags_unknown = []

        all_right_tag_list = []
        all_right_tag_list_known = []
        all_right_tag_list_unknown = []

        for num, sentence in enumerate(sentence_list):
            if num + 1 % 10 == 0:
                print(f'handling sentence number {num + 1}')
            cur_res = self.predict(sentence)
            cur_res_known = [tag_res for hist, tag_res in zip(sentence, cur_res) if
                             self.word_possible_tag_set.get(hist.cword, None)]
            cur_res_unknown = [tag_res for hist, tag_res in zip(sentence, cur_res) if
                               not self.word_possible_tag_set.get(hist.cword, None)]
            all_res_tags.append(cur_res)

            ground_truth = [hist.ctag for hist in sentence]
            ground_truth_known = [hist.ctag for hist in sentence if self.word_possible_tag_set.get(hist.cword, None)]
            ground_truth_unknown = [hist.ctag for hist in sentence if
                                    not self.word_possible_tag_set.get(hist.cword, None)]
            all_gt_tags.append(ground_truth)
            all_gt_tags_known.append(ground_truth_known)
            all_gt_tags_unknown.append(ground_truth_unknown)
            res_acc, right_tag_list = self.calc_accuracy(cur_res, ground_truth)
            res_acc_known, right_tag_list_known = self.calc_accuracy(cur_res_known, ground_truth_known)
            res_acc_unknown, right_tag_list_unknown = self.calc_accuracy(cur_res_unknown, ground_truth_unknown)
            if num + 1 % 10 == 0:
                print(f'accuracy for sentence {num + 1}: {res_acc} \n'
                      f'known words accuracy for sentence {num + 1}: {res_acc_known}'
                      f'unknown words accuracy for sentence {num + 1}: {res_acc_unknown}')

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
        q.put((i, all_res_tags,
               all_acc_list,
               all_tagged_res_list,
               all_gt_tags,
               all_gt_tags_known,
               all_gt_tags_unknown,
               all_right_tag_list,
               all_right_tag_list_known,
               all_right_tag_list_unknown))

    def predict_all_test(self, num_workers=4):
        p = mp.Pool(num_workers)
        manager = mp.Manager()
        q = manager.Queue()
        args = self._prep_args(num_workers=num_workers, q=q)
        p.starmap(func=self._calc_predict, iterable=args)
        p.close()
        p.join()
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

        res_list = []
        while q.qsize() > 0:
            res_list.append(q.get())
        # sort to make sure that the results are in correct order
        res_list = list(sorted(res_list, key=lambda x: x[0]))

        # update results from all processes
        for _, all_res_tags_cur, \
            all_acc_list_cur, \
            all_tagged_res_list_cur, \
            all_gt_tags_cur, \
            all_gt_tags_known_cur, \
            all_gt_tags_unknown_cur, \
            all_right_tag_list_cur, \
            all_right_tag_list_known_cur, \
            all_right_tag_list_unknown_cur in res_list:


            all_res_tags += all_res_tags_cur
            all_acc_list += all_acc_list_cur
            all_tagged_res_list += all_tagged_res_list_cur
            all_gt_tags += all_gt_tags_cur
            all_gt_tags_known += all_gt_tags_known_cur
            all_gt_tags_unknown += all_gt_tags_unknown_cur
            all_right_tag_list += all_right_tag_list_cur
            all_right_tag_list_known += all_right_tag_list_known_cur
            all_right_tag_list_unknown += all_right_tag_list_unknown_cur


        print(f'precent of known words in corpus: {100 * len(all_right_tag_list_known)/len(all_right_tag_list)}')
        print(f'precent of unknown words in corpus: {100 * len(all_right_tag_list_unknown) / len(all_right_tag_list)}')
        self.total_acc = self._calc_acc(all_right_tag_list)
        print(f'total accuracy: {self.total_acc}')
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
        tag_set = self.tags_list[:-2]
        return tag_set

    # @timeit
    def calc_prob_for_hist(self, cur_hist, prev_pi, u):
        pptag_len = len(prev_pi)
        cur_pi = []
        for pptag in range(pptag_len):
            if prev_pi[pptag] == 0:
                cur_pi.append(np.zeros(len(self.tags_list),dtype=np.float64))
                continue
            pptag = self.index_to_tag[pptag]
            dots = []
            for c_tag in self.get_possible_tag_set_from_word(cur_hist.cword):
                n_hist = History(cword=cur_hist.cword, pptag=pptag, ptag=u,
                                 nword=cur_hist.nword, pword=cur_hist.pword, ctag=c_tag,
                                 nnword=cur_hist.nnword, ppword=cur_hist.ppword)
                dot_prod = self.v.dot(self.get_feature_from_hist(n_hist))
                dots.append(dot_prod)
            exp_arr = np.exp(np.array(dots))
            prob_arr = exp_arr / np.sum(exp_arr)
            prob_arr = np.append(np.append(prob_arr,0),0)
            cur_pi.append(prev_pi[self.tag_to_index[pptag]]*prob_arr)
        return np.array(cur_pi)

    def calc_res_tags(self, sentence):
        beam_set = {'*'}
        for ind, k in enumerate(range(1, len(sentence) + 1)):
            cur_hist = sentence[ind]
            for u in beam_set:
                prev_pi = self.pi_tables[k-1, :, self.tag_to_index[u]]
                probs = self.calc_prob_for_hist(cur_hist, prev_pi, u)
                self.pi_tables[k, self.tag_to_index[u], :] = np.max(probs, axis=0)
                self.bp_tables[k, self.tag_to_index[u], :] = np.argmax(probs, axis=0)

            beam_idxs = np.argsort(self.pi_tables[k, [self.tag_to_index[u] for u in beam_set], :].flatten())
            beam_idxs = np.flip(beam_idxs % self.pi_tables.shape[-1])
            beam_idxs = beam_idxs[:self.beam_width ** 2]
            beam_set = set()
            for i,tag_idx in enumerate(beam_idxs):
                beam_set.add(self.index_to_tag[beam_idxs[i]])
                if len(beam_set) == self.beam_width:
                    break

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
        self.pi_tables = np.full(shape=(len(sentence) + 1, len(self.tags_list), len(self.tags_list)), fill_value=0.)
        self.pi_tables[0, self.tag_to_index["**"], self.tag_to_index["*"]] = 1.
        self.bp_tables = np.zeros(shape=self.pi_tables.shape, dtype=np.int)
        # self.fill_prob_dict_from_sentence(sentence)
        res_tags = self.calc_res_tags(sentence)
        return res_tags

    def dump_res(self, all_tagged_res_list, all_gt_tags, all_res_tags, sentence_list):
        if not os.path.isdir(self.res_path):
            os.makedirs(self.res_path)
        dump_name = self.dump_name + '_threshold_' + str(self.threshold) + '_lambda_' + str(self.reg_lambda)\
                    + '_beam_' + str(self.beam_width) + '_acc_' + str(self.total_acc)
        dump_path = os.path.join(self.res_path, dump_name)
        with open(dump_path, 'wb') as f:
            res = (all_tagged_res_list, all_gt_tags, all_res_tags, sentence_list)
            pickle.dump(res, f)
