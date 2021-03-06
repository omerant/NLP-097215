import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from re import split


class ResultsHandler:
    def __init__(self):
        self.res_path = '../res'
        self.dump_name = 'test1_threshold_3_lambda_0.7_beam_2_acc_95.5267381937991'
        self.cf_name = 'cf'

    def get_res(self):
        dump_path = os.path.join(self.res_path, self.dump_name)
        with open(dump_path, 'rb') as f:
            self.all_tagged_res_list, self.all_gt_tags, self.all_res_tags, self.sentence_list = pickle.load(f)
        self.process_results()

    def process_results(self):
        self.y_true = [label for sentence in self.all_gt_tags for label in sentence]
        self.y_pred = [label for sentence in self.all_res_tags for label in sentence]
        self.all_labels = list(dict.fromkeys(self.y_true))
        self.idx_to_labels = {idx: label for idx,label in enumerate(self.all_labels)}
        self.label_to_idx = {label: idx for idx, label in self.idx_to_labels.items()}
        self.words = [wordtag.split('_')[0] for sentence in self.all_tagged_res_list for wordtag in sentence]
        self.true_tags = []

    def plot_confusion_matrix(self):
        self.C = confusion_matrix(self.y_true, self.y_pred, labels=self.all_labels)
        acc = np.trace(self.C) / np.sum(self.C)
        print(acc)
        df_cm = pd.DataFrame(self.C, self.all_labels, self.all_labels)
        sn.set(font_scale=0.3)
        hm = sn.heatmap(df_cm,annot=True, robust=True)
        save_path = os.path.join(self.res_path, self.cf_name)
        cm = hm.get_figure()
        cm.savefig(save_path, dpi=400)

    def get_most_mistakes(self):
        mistakes = {}
        for word,pred,true in zip(self.words,self.y_pred,self.y_true):
            if pred != true:
                if not mistakes.get(word,None):
                    mistakes[word] = [(pred,true)]
                else:
                    mistakes[word].append((pred,true))
        most_mistakes = {k:v for k,v in sorted(mistakes.items(), key=lambda item:len(item[1]),reverse=True)}
        word_occurrences = {}
        for word in self.words:
            if not word_occurrences.get(word,None):
                word_occurrences[word] = 1
            else:
                word_occurrences[word]+=1
        mistakes_ratio = {k:len(v)/word_occurrences[k] for k,v in most_mistakes.items() if len(v)>10}
        mistakes_ratio = {k:v for k,v in sorted(mistakes_ratio.items(), key=lambda item:item[1], reverse=True)}
        common_words_mistakes = {k:set(most_mistakes[k]) for k,v in mistakes_ratio.items()}
        most_mistakes_num = {k:len(v) for k,v in most_mistakes.items() if len(v)>10}

        print('Ratio of mistakes per word:')
        print(mistakes_ratio)
        print('(predicted tag, true tag) for all different mistakes per word:')
        print(common_words_mistakes)
        print('Num of mistakes per word:')
        print(most_mistakes_num)

    def acc_per_label(self):
        accs = {self.idx_to_labels[i]: (row[i] / np.sum(row)) if np.sum(row)>0 else 0 for i,row in enumerate(self.C)}
        print("% acc per real label: ", accs)
        worst_accs = [(k, v) for k, v in sorted(accs.items(), key=lambda item: item[1])]
        print("worst 5: ", worst_accs[:5])
        accs = {self.idx_to_labels[i]: (col[i] / np.sum(col)) if np.sum(col)>0 else 0 for i,col in enumerate(self.C.T)}
        print("% acc per predicted label:", accs)
        worst_accs = [(k,v) for k,v in sorted(accs.items(), key=lambda item:item[1])]
        print("worst 5: ", worst_accs[:5])

    def new_words(self):
        train1_path = 'data/train1.wtag'
        all_train_words = []
        with open(train1_path) as f:
            for idx, line in enumerate(f):
                splited_words = split(' |,\n', line[:-1]) if line[-1] == '\n' else split(' |,\n', line)  # remove \n from last part of sentence
                for word_idx in range(len(splited_words)):
                    cword, ctag = split('_', splited_words[word_idx])
                    all_train_words.append(cword)
        test1_path = 'data/test1.wtag'
        all_test_words = []
        with open(test1_path) as f:
            for idx, line in enumerate(f):
                splited_words = split(' |,\n', line[:-1]) if line[-1] == '\n' else split(' |,\n', line)  # remove \n from last part of sentence
                for word_idx in range(len(splited_words)):
                    cword, ctag = split('_', splited_words[word_idx])
                    all_test_words.append(cword)
        new_words=[word for word in all_test_words if word not in all_train_words]
        print(len(new_words))

    def create_conf_mat(self):
        conf = np.zeros((len(self.all_labels), len(self.all_labels)))
        for pred, true in zip(self.y_pred, self.y_true):
            conf[self.label_to_idx[true], self.label_to_idx[pred]] +=1
        conf_tmp = np.copy(conf)
        conf_tmp[np.diag_indices_from(conf_tmp)] = 0
        top_ten_mistakes = np.argsort(-conf_tmp.sum(axis=1))[:10]
        top_ten_mistakes_labels = [self.idx_to_labels[idx] for idx in top_ten_mistakes]
        conf_tmp = conf[top_ten_mistakes][:,top_ten_mistakes]
        fig, axs = plt.subplots(2, 1)
        axs[0].axis('tight')
        axs[0].axis('off')
        axs[0].table(conf_tmp, rowLabels=top_ten_mistakes_labels, colLabels=top_ten_mistakes_labels, loc='center')
        plt.show()


#
res = ResultsHandler()
# res.new_words()
res.get_res()
res.create_conf_mat()
# res.plot_confusion_matrix()
# res.get_most_mistakes()
# res.acc_per_label()
