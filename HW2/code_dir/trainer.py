import time
import os
import torch
from torch.utils.data.dataloader import DataLoader


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
                # print("pos_idx_tensor shape -", pos_idx_tensor.shape)
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
            train_accuracy = self.calculate_accuracy(self.model, dl_train, len_train, self.device)
            train_acc_list.append(train_accuracy)
            cur_epoch_val_accuracy = self.calculate_accuracy(self.model, dl_val, len_test, self.device)
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
                }
                print('saving model')
                torch.save(state, 'checkpoints/' + self.model.name +  '.pth')

        print('==> Finished Training ...')

    def evaluate(self, test_dataloader, len_test):
        acc = 0
        with torch.no_grad():
            for batch_idx, input_data in enumerate(test_dataloader):
                words_idx_tensor, pos_idx_tensor, sentence_length = input_data
                tag_scores = self.model(words_idx_tensor)
                tag_scores = tag_scores.unsqueeze(0).permute(0, 2, 1)

                _, indices = torch.max(tag_scores, 1)
                acc += torch.mean(torch.tensor(pos_idx_tensor.to("cpu") == indices.to("cpu"), dtype=torch.float))
            acc = acc / len_test
        return acc

    @staticmethod
    def calculate_accuracy(model, dataloader, len_data, device):
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
