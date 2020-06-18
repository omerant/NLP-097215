import torch
import torch.nn.functional as F
from torch.nn import NLLLoss


class NllLoss(torch.nn.Module):

    def __init__(self):
        super(NllLoss, self).__init__()

    def forward(self, log_prob_y_pred, y_true) -> torch.Tensor:
        """

        :param y_true:
        :param y_pred:
        :return:
        """

        Y_i_size = y_true.shape[1]
        our_loss = -(1. / (Y_i_size * y_true.shape[0])) * torch.sum(log_prob_y_pred.gather(1, y_true.unsqueeze(1)))
        return our_loss


class HingeLoss(torch.nn.Module):

    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, y_pred, y_true) -> torch.Tensor:
        """

        :param y_true: 2d tensor
        :param y_pred: 1d tensor
        :return:
        """
        # print(f'y_pred shape: {y_pred.shape}')
        # print(f'y_true shape: {y_true.shape}')
        # print(f'y_pred.requires_grad: {y_pred.requires_grad}')
        y_pred = y_pred.T.squeeze(0)
        y_true = y_true.squeeze(0)
        mask = torch.ones(y_pred.shape, dtype=torch.bool)
        mask[torch.LongTensor(range(y_true.shape[0])).to(y_pred.device), y_true] = False
        score_max_incorrect_tensor = y_pred[mask].view(y_pred.shape[0], -1).max(dim=1)[0] + 1 if y_pred[mask].shape[0] > 1 else y_pred[mask] + 1
        score_correct_tensor = y_pred[~mask]
        res = torch.max(torch.tensor([0], dtype=torch.float, requires_grad=True).to(y_pred.device),
                        (1 + score_max_incorrect_tensor.sum() - score_correct_tensor.sum()).float())[0]
        return res


if __name__ == '__main__':

    # # test loss with 3d tensor
    # y_pred = torch.tensor(([[0.88, 0.12], [0.51, 0.49]], [[0.88, 0.12], [0.51, 0.49]], [[0.88, 0.12], [0.51, 0.49]]),
    #                       dtype=torch.float)
    # y_true_out = torch.tensor([[1, 0], [1, 0], [1, 0]])
    #
    # # y_pred.requires_grad = True
    # loss_fn = NllLoss()
    # log_prob_y_pred_out = F.log_softmax(y_pred, 2)
    # my_loss = loss_fn(log_prob_y_pred_out[:, :, 1:], y_true_out[:, 1:])
    # print(my_loss)
    # # tensor(0.5718)
    # loss_fn = NLLLoss()
    # lib_loss = loss_fn(log_prob_y_pred_out[:, :, 1:], y_true_out[:, 1:])#loss_fn(log_prob_y_pred_out, y_true_out)
    # print(lib_loss)
    # assert lib_loss == my_loss

    y_pred = torch.tensor([[1., 2], [3, 7], [4, 17]], requires_grad=True).unsqueeze(0)
    y_true = torch.LongTensor([0., 0]).unsqueeze(0)

    # y_pred = torch.tensor([[1., 2]], requires_grad=True).unsqueeze(0)
    # y_true = torch.LongTensor([0.]).unsqueeze(0)
    h_loss = HingeLoss()
    loss = h_loss(y_pred, y_true)
    loss.backward()
    print(loss)

