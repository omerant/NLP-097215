import time
import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from constants import UNK_IDX
from utils import constract_line, get_vocabs_dep_parser
from data_handling import DepDataset
from model import DnnSepParser


class Trainer:

    def __init__(self, model, optimizer, loss, device):
        self.model = model
        self.loss_fn = loss
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs: int, dl_train: DataLoader, dl_val: DataLoader, acumulate_grad_steps: int,
              len_train: int, len_test: int, early_stopping: bool = None):
        epochs_without_improvement = 0
        train_acc_list = []
        val_acc_list = []
        best_val_acc = 0.
        for epoch in range(1, num_epochs + 1):
            self.model.train()  # put in training mode
            running_epoch_loss = 0.0
            epoch_time = time.time()
            i = 0
            for data in dl_train:
                i += 1
                # get the inputs
                words_idx_tensor, pos_idx_tensor, sentence_length = data
                tag_scores = self.model(words_idx_tensor)
                tag_scores = tag_scores.unsqueeze(0).permute(0, 2, 1)
                # print("tag_scores shape -", tag_scores.shape)
                # print(f'tag_scores: {tag_scores}')
                # print("pos_idx_tensor shape -", pos_idx_tensor.shape)
                # print(f'pos_idx_tensor: {pos_idx_tensor}')
                loss = self.loss_fn(tag_scores, pos_idx_tensor.to(self.device))
                loss = loss / acumulate_grad_steps
                loss.backward()
                if i % acumulate_grad_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                running_epoch_loss += loss.data.item()
            # Normalizing the loss by the total number of train batches
            running_epoch_loss /= len_train
            # Calculate training/test set accuracy of the existing model
            train_accuracy = self.calculate_accuracy_pos_tagging(self.model, dl_train, len_train, self.device)
            train_acc_list.append(train_accuracy)
            cur_epoch_val_accuracy = self.calculate_accuracy_pos_tagging(self.model, dl_val, len_test, self.device)
            val_acc_list.append(cur_epoch_val_accuracy)
            log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(epoch,
                                                                                                             running_epoch_loss,
                                                                                                             train_accuracy,
                                                                                                             cur_epoch_val_accuracy)
            epoch_time = time.time() - epoch_time
            log += "Epoch Time: {:.2f} secs".format(epoch_time)
            print(log)

            if len(va) > 1 and cur_epoch_val_accuracy < val_acc_list[-2]:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0

            if early_stopping is not None and epochs_without_improvement == early_stopping:
                break

            if cur_epoch_val_accuracy > best_val_acc:
                best_val_acc = cur_epoch_val_accuracy
                if not os.path.isdir('checkpoints'):
                    os.mkdir('checkpoints')
                # save model
                state = {
                    'net': self.model.state_dict(),
                    'epoch': epoch,
                    'val_acc_list': val_acc_list,
                    'train_acc_list': train_acc_list
                }
                print('saving model')
                torch.save(state, 'checkpoints/' + self.model.name +  '.pth')

        print('==> Finished Training ...')

    def train_dep_parser(self, num_epochs: int, dl_train: DataLoader, dl_val: DataLoader, acumulate_grad_steps: int,
                         len_train: int, len_test: int, early_stopping: int = None):
        epochs_without_improvement = 0
        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list = []
        best_val_loss = np.inf
        for epoch in range(1, num_epochs + 1):
            self.model.train()  # put in training mode
            running_epoch_loss = 0.0
            running_epoch_acc = []
            epoch_time = time.time()
            i = 0
            for data in dl_train:
                i += 1
                if i % 1000 == 0:
                    print(i)
                word_dropout_prob, words_idx_tensor, pos_idx_tensor, dep_idx_tensor, sentence_length = data
                # insert dropout
                bern_distribution = torch.distributions.bernoulli.Bernoulli(word_dropout_prob)
                words_idx_tensor[bern_distribution.sample().bool()] = UNK_IDX
                # print(f'UNK_IDX: {UNK_IDX}')
                # print(f'words_idx_tensor: {words_idx_tensor}')
                dep_scores, our_heads, scores = self.model(words_idx_tensor, pos_idx_tensor, True)
                dep_scores = dep_scores.unsqueeze(0).permute(0, 2, 1)

                dep_idx_tensor_2d = dep_idx_tensor.squeeze(0)
                running_epoch_acc += [np.mean(dep_idx_tensor_2d.numpy()[1:] == our_heads[1:])]

                # CALC LOSS ON SCORES
                # scores = scores.unsqueeze(0).permute(0, 2, 1)
                # loss = self.loss_fn(scores[:, :, 1:], dep_idx_tensor.to(self.device)[:, 1:])
                # print(f'loss.device: {loss.device}')
                # END CALC LOSS ON SCORES
                # print(f'dep_idx_tensor.to(self.device)[:, 1:]: {dep_idx_tensor.to(self.device)[:, 1:]}')
                loss = self.loss_fn(dep_scores[:, :, 1:], dep_idx_tensor.to(self.device)[:, 1:])
                loss = loss / acumulate_grad_steps
                loss.backward()
                if i % acumulate_grad_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()

                running_epoch_loss += loss.data.item()
            # Normalizing the loss by the total number of train batches
            running_epoch_loss /= (len_train/acumulate_grad_steps)
            train_loss_list.append(running_epoch_loss)
            # print(f'running_epoch_acc: {running_epoch_acc}')
            cur_epoch_train_accuracy = np.mean(running_epoch_acc) * 100
            train_acc_list.append(cur_epoch_train_accuracy)
            cur_epoch_val_accuracy, cur_epoch_val_loss = self.calculate_accuracy_dep_parser(self.model, dl_val,
                                                                                            len_test, self.loss_fn,
                                                                                            self.device, acumulate_grad_steps)

            val_loss_list.append(cur_epoch_val_loss)
            val_acc_list.append(cur_epoch_val_accuracy)
            log = "Epoch: {} | Training Loss: {:.4f} | Training accuracy: {:.3f}% | Test Loss: {:.4f} | Test accuracy: {:.3f}% | ".format(epoch,
                                                                                                             running_epoch_loss,
                                                                                                             cur_epoch_train_accuracy,
                                                                                                             cur_epoch_val_loss,
                                                                                                             cur_epoch_val_accuracy)
            epoch_time = time.time() - epoch_time
            log += "Epoch Time: {:.2f} secs".format(epoch_time)
            print(log)

            if len(val_loss_list) > 1 and cur_epoch_val_loss > best_val_loss:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0

            if early_stopping is not None and epochs_without_improvement == early_stopping:
                print(f'early stopping reached, stop training')
                break

            if cur_epoch_val_loss < best_val_loss:
                best_val_loss = cur_epoch_val_loss
                if not os.path.isdir('checkpoints'):
                    os.mkdir('checkpoints')
                # save model
                state = {
                    'net': self.model.state_dict(),
                    'epoch': epoch,
                    'val_loss_list': val_loss_list,
                    'val_acc_list': val_acc_list,
                    'train_loss_list': train_loss_list,
                    'train_acc_list': train_acc_list,
                    'word_embedding': self.model.word_embedding,
                    'tag_embedding': self.model.tag_embedding
                }
                print('saving model')
                # insert more parameters to save
                torch.save(state, 'checkpoints/' + self.model.name + '_' + str(acumulate_grad_steps) + '.pth')

        print('==> Finished Training ...')

    @staticmethod
    def calculate_accuracy_pos_tagging(model, dataloader, len_data, device):
        acc = 0
        with torch.no_grad():
            for batch_idx, input_data in enumerate(dataloader):
                words_idx_tensor, pos_idx_tensor, sentence_length = input_data
                tag_scores = model(words_idx_tensor)
                tag_scores = tag_scores.unsqueeze(0).permute(0, 2, 1)

                _, indices = torch.max(tag_scores, 1)
                acc += torch.mean(torch.tensor(pos_idx_tensor.to("cpu") == indices.to("cpu"), dtype=torch.float))
            acc = acc / len_data
            acc *= 100
        return acc

    @staticmethod
    def calculate_accuracy_dep_parser(model, dataloader, len_data, loss_fn, device, acumulate_grad_steps):
        acc_list = []
        unknow_acc_list = []
        loss_list = []
        non_skip_idx = 1
        with torch.no_grad():
            # count = 0
            for batch_idx, input_data in enumerate(dataloader):
                if batch_idx % non_skip_idx == 0:
                    word_dropout_prob, words_idx_tensor, pos_idx_tensor, dep_idx_tensor, sentence_length = input_data

                    dep_scores, our_heads, scores = model(words_idx_tensor, pos_idx_tensor, calc_mst=True)
                    assert our_heads is not None
                    dep_scores = dep_scores.unsqueeze(0).permute(0, 2, 1)
                    dep_idx_tensor_2d = dep_idx_tensor.squeeze(0)
                    acc_list += [np.mean(dep_idx_tensor_2d.numpy()[1:] == our_heads[1:])]
                    unk_idx_arr = words_idx_tensor.squeeze(0)[1:] == UNK_IDX
                    if sum(unk_idx_arr) > 0:

                        unknow_acc_list += [np.mean(dep_idx_tensor_2d.numpy()[1:][unk_idx_arr] == our_heads[1:][unk_idx_arr])]

                    # CALC LOSS ON SCORES
                    # scores = scores.unsqueeze(0).permute(0, 2, 1)
                    # cur_loss = loss_fn(scores[:, :, 1:], dep_idx_tensor.to(device)[:, 1:])
                    # END CALC LOSS ON SCORES

                    loss = loss_fn(dep_scores[:, :, 1:], dep_idx_tensor.to(device)[:, 1:])





                    loss_list.append(loss.cpu().numpy()/acumulate_grad_steps)

            acc = np.mean(acc_list)
            acc *= 100
            # print(f'acc list: P{acc_list}')
            unknown_acc = np.mean(unknow_acc_list) * 100
            print(f'unknown accuracy: {unknown_acc}')
            loss = np.mean(loss_list) * acumulate_grad_steps
        return acc, loss


if __name__ == '__main__':
    path_train = "data_new/train.labeled"
    path_test = "data_new/test.labeled"
    paths_list_train = [path_train]
    word_dict_train, pos_dict_train = get_vocabs_dep_parser(paths_list_train)

    paths_list_all = [path_train, path_test]
    word_dict_all, pos_dict_all = get_vocabs_dep_parser(paths_list_all)

    train = DepDataset(word_dict_all, pos_dict_all, 'data_new', 'train', padding=False)
    train_dataloader = DataLoader(train, shuffle=True)
    test = DepDataset(word_dict_all, pos_dict_all, 'data_new', 'test', padding=False, train_word_dict=word_dict_all)
    test_dataloader = DataLoader(test, shuffle=False)

    word_vocab_size = len(train.word_idx_mappings)
    tag_vocab_size = len(train.pos_idx_mappings)
    # init model
    model = DnnSepParser(word_emb_dim=100, tag_emb_dim=25, num_layers=2, word_vocab_size=word_vocab_size,
                         tag_vocab_size=tag_vocab_size, hidden_fc_dim=100)
    # load weights
    path = 'checkpoints/first_model/DnnDepParser_word_emb-100_tag_emb-25_num_stack2_20.pth'
    train_info = torch.load(path, map_location=torch.device('cpu'))
    word_emb = train_info['word_embedding']
    tag_emb = train_info['tag_embedding']
    model.load_state_dict(train_info['net'])
    model.load_embedding(word_emb, tag_emb)
    Trainer.calculate_comp(model=model, comp_dataloader=test_dataloader, idx_word_mappings=test.idx_word_mappings,
                           idx_pos_mappings=test.idx_pos_mappings, output_path='beza')
    pass
