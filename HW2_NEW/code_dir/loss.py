import torch
import torch.nn.functional as F


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
        our_loss = -(1. / (Y_i_size * y_true.shape[0])) * torch.sum(log_prob_y_pred[:,:,1:].gather(1, y_true.unsqueeze(1)[:,:,1:]))
        return our_loss


if __name__ == '__main__':

    # test loss with 3d tensor
    y_pred = torch.tensor(([[0.88, 0.12], [0.51, 0.49]], [[0.88, 0.12], [0.51, 0.49]], [[0.88, 0.12], [0.51, 0.49]]),
                          dtype=torch.float)
    y_true_out = torch.tensor([[1, 0], [1, 0], [1, 0]])

    # y_pred.requires_grad = True
    loss_fn = NllLoss()
    log_prob_y_pred_out = F.log_softmax(y_pred, 2)
    my_loss = loss_fn(log_prob_y_pred_out, y_true_out)
    print(my_loss)

    loss_fn = torch.nn.NLLLoss()
    lib_loss = loss_fn(log_prob_y_pred_out, y_true_out)
    print(lib_loss)
    assert lib_loss == my_loss

    # TEST IGNORE INDEX
    y_pred_test = torch.tensor(([[0.88, 0.12], [0.51, 0.49]], [[0.88, 0.12], [0.51, 0.49]], [[0.88, 0.12], [0.51, 0.49]]),
                          dtype=torch.float)
    y_true_test = torch.tensor([[0, -20], [-20, -20], [-20, -20]])

    loss_fn = torch.nn.NLLLoss(ignore_index=-20)
    log_prob_y_pred_out = F.log_softmax(y_pred_test, 2)
    ignore_index_loss = loss_fn(log_prob_y_pred_out, y_true_test)
    print(f'with ignore index: {ignore_index_loss}')
