import torch
import matplotlib.pyplot as plt


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


checkpoint_path = 'checkpoints/DnnDepParser_word_emb-100_tag_emb-25_num_stack3.pth'
plot_train(checkpoint_path)
