import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

class ResultsHandler:
    def __init__(self):
        self.res_path = 'res'
        self.dump_name = 'test1'
        self.cf_name = 'cf'

    def get_res(self):
        dump_path = os.path.join(self.res_path, self.dump_name)
        with open(dump_path, 'rb') as f:
            self.all_tagged_res_list, self.all_gt_tags, self.all_res_tags = pickle.load(f)
        self.process_results()

    def process_results(self):
        self.y_true = [label for sentence in self.all_gt_tags for label in sentence]
        self.y_pred = [label for sentence in self.all_res_tags for label in sentence]
        self.all_labels = list(dict.fromkeys(self.y_pred))
        self.idx_to_labels = {idx: label for idx,label in enumerate(self.all_labels)}


    def plot_confusion_matrix(self):
        self.C = confusion_matrix(self.y_true, self.y_pred)
        acc = np.trace(self.C) / np.sum(self.C)
        print(acc)
        df_cm = pd.DataFrame(self.C, self.all_labels, self.all_labels)
        sn.set(font_scale=0.3)
        hm = sn.heatmap(df_cm,annot=True,robust=True)
        save_path = os.path.join(self.res_path, self.cf_name)
        cm = hm.get_figure()
        cm.savefig(save_path, dpi=400)

    def get_most_mistakes(self):
        mistakes = self.C
        mistakes[np.diag_indices_from(mistakes)] = 0
        max_idxs = np.argpartition(mistakes,-10, axis=None)[-10:]
        print(max_idxs)
        print(mistakes[max_idxs])

    def acc_per_label(self):
        accs = {self.idx_to_labels[i]: (row[i] / np.sum(row)) if np.sum(row)>0 else 0 for i,row in enumerate(self.C)}
        print("% acc per real label: ", accs)
        worst_accs = [(k, v) for k, v in sorted(accs.items(), key=lambda item: item[1])]
        print("worst 5: ", worst_accs[:5])
        accs = {self.idx_to_labels[i]: (col[i] / np.sum(col)) if np.sum(col)>0 else 0 for i,col in enumerate(self.C.T)}
        print("% acc per predicted label:", accs)
        worst_accs = [(k,v) for k,v in sorted(accs.items(), key=lambda item:item[1])]
        print("worst 5: ", worst_accs[:5])


#
res = ResultsHandler()
res.get_res()
res.plot_confusion_matrix()
# res.get_most_mistakes()
res.acc_per_label()
