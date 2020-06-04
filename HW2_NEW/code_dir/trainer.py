import time
import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from chu_liu_edmonds import decode_mst


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

            if len(val_acc_list) > 1 and cur_epoch_val_accuracy < val_acc_list[-2]:
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
                words_idx_tensor, pos_idx_tensor, dep_idx_tensor, sentence_length = data
                dep_scores = self.model(words_idx_tensor, pos_idx_tensor)
                dep_scores = dep_scores.unsqueeze(0).permute(0, 2, 1)
                # print("tag_scores shape -", tag_scores.shape)
                # print(f'tag_scores: {tag_scores}')
                # print("pos_idx_tensor shape -", pos_idx_tensor.shape)
                # print(f'pos_idx_tensor: {pos_idx_tensor}')
                loss = self.loss_fn(dep_scores, dep_idx_tensor.to(self.device))
                loss = loss / acumulate_grad_steps
                loss.backward()
                if i % acumulate_grad_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                running_epoch_loss += loss.data.item()
            # Normalizing the loss by the total number of train batches
            running_epoch_loss /= len_train
            # Calculate training/test set accuracy of the existing model
            train_accuracy = self.calculate_accuracy_dep_parser(self.model, dl_train, len_train, self.device)
            train_acc_list.append(train_accuracy)
            cur_epoch_val_accuracy = self.calculate_accuracy_dep_parser(self.model, dl_val, len_test, self.device)
            val_acc_list.append(cur_epoch_val_accuracy)
            log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(epoch,
                                                                                                             running_epoch_loss,
                                                                                                             train_accuracy,
                                                                                                             cur_epoch_val_accuracy)
            epoch_time = time.time() - epoch_time
            log += "Epoch Time: {:.2f} secs".format(epoch_time)
            print(log)

            if len(val_acc_list) > 1 and cur_epoch_val_accuracy < val_acc_list[-2]:
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
    def calculate_accuracy_dep_parser(model, dataloader, len_data, device):
        acc = 0
        with torch.no_grad():
            for batch_idx, input_data in enumerate(dataloader):
                words_idx_tensor, pos_idx_tensor, dep_idx_tensor, sentence_length = input_data
                dep_scores = model(words_idx_tensor, pos_idx_tensor)
                dep_scores = dep_scores.unsqueeze(0).permute(0, 2, 1)
                dep_scores = dep_scores.squeeze(0)
                # print(f'dep_scores shape: {dep_scores.shape}')
                our_heads, _ = decode_mst(energy=dep_scores.cpu(), length=sentence_length, has_labels=False)
                # _, indices = torch.max(dep_scores, 1)
                # TODO: fix acc calculation according to UAS
                dep_idx_tensor = dep_idx_tensor.squeeze(0)
                # print(f'dep_idx_tensor shape: {dep_idx_tensor.shape}')
                # print(f'dep_idx_tensor: {dep_idx_tensor}')
                # print(f'our_heads shape: {our_heads.shape}')
                # print(f'our_heads: {our_heads}')
                # print(f'type our_heads: {type(our_heads)}')
                # print(f'dep_idx_tensor.numpy() == our_heads: {dep_idx_tensor.numpy() == our_heads}')
                acc += np.mean(dep_idx_tensor.numpy() == our_heads)
                # print(f'acc: {acc}')
            acc = acc / len_data
            acc *= 100
        return acc


if __name__ == '__main__':
    dep_scores_example = torch.Tensor([[], []])