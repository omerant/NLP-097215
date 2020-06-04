import torch
import torch.nn.functional as F


class NllLoss(torch.nn.Module):

    def __init__(self):
        super(NllLoss, self).__init__()

    def forward(self, y_true, y_pred) -> torch.Tensor:
        """

        :param y_true:
        :param y_pred:
        :return:
        """
        Y_i_size = y_true.shape[1]
        prob_y_pred = F.softmax(y_pred, 2)
        our_loss = -(1. / (Y_i_size * y_true.shape[0])) * torch.sum(torch.log(prob_y_pred.gather(1, y_true.unsqueeze(1))))
        return our_loss


if __name__ == '__main__':

    # test loss with 3d tensor
    y_pred = torch.tensor(([[0.88, 0.12], [0.51, 0.49]], [[0.88, 0.12], [0.51, 0.49]], [[0.88, 0.12], [0.51, 0.49]]), dtype=torch.float)
    y_true = torch.tensor([[1, 0], [1, 0], [1, 0]])

    # y_pred.requires_grad = True
    loss_fn = NllLoss()
    my_loss = loss_fn(y_true, y_pred)
    print(my_loss)
    prob_y_pred = F.softmax(y_pred, 2)

    loss_fn = torch.nn.NLLLoss(reduction='mean')
    lib_loss = loss_fn(torch.log(prob_y_pred), y_true)
    print(lib_loss)
    assert lib_loss == my_loss


