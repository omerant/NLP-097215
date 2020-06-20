import torch
import matplotlib.pyplot as plt
import os


def plot_train(path):
    train_info = torch.load(path, map_location=torch.device('cpu'))
    val_loss_list = train_info['val_loss_list']
    val_acc_list = train_info['val_acc_list']

    train_loss_list = train_info['train_loss_list']
    train_acc_list = train_info['train_acc_list']

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(val_loss_list, 'tab:orange')
    axs[0, 0].set_title('Validation Loss')
    axs[0, 1].plot(val_acc_list, 'tab:orange')
    axs[0, 1].set_title('Validation UAS')
    axs[1, 0].plot(train_loss_list)
    axs[1, 0].set_title('Train Loss')
    axs[1, 1].plot(train_acc_list)
    axs[1, 1].set_title('Train UAS')
    plt.show()
    pass


def get_model_with_highest_val_acc(dir_path):
    best_val_acc = 0
    train_acc = None
    best_file = None
    for file in sorted(os.listdir(dir_path)):
        print(file)
        path = os.path.join(dir_path, file)
        train_info = torch.load(path, map_location=torch.device('cpu'))
        val_acc_list = train_info['val_acc_list']
        cur_best_val_acc = val_acc_list[-1]
        if cur_best_val_acc > best_val_acc:
            best_val_acc = cur_best_val_acc
            best_file = file
            train_acc = train_info['train_acc_list'][-1]
    return best_file, best_val_acc, train_acc


checkpoint_path = 'checkpoints/best_second_model'
best_model_file, best_val_acc, train_acc = get_model_with_highest_val_acc(checkpoint_path)
print(f'best validation accuracy: {best_val_acc}')
print(f'best train accuracy: {train_acc}')
print(f'best_model_file: {best_model_file}')
model_param_path = os.path.join(checkpoint_path, best_model_file)
plot_train(model_param_path)
