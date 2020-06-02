import torch


class NllLoss(torch.nn.Module):

    def __init__(self):
        super(NllLoss, self).__init__()

    def forward(self, y_true, y_pred)->torch.Tensor:
        Y_i_size = torch.sum(y_true == True).item()
        y_pred_exp = torch.exp(y_pred)
        prob_y_pred = y_pred_exp / y_pred_exp.sum(dim=0)
        loss = -(1. / Y_i_size) * torch.sum(torch.log(prob_y_pred[torch.where(y_true)]))
        return loss


if __name__ == '__main__':
    y_true = torch.LongTensor([[False, True, False], [False, False, True], [False, False, False]])
    y_pred = torch.Tensor([[0, 1, 2.], [1, 0, 0.5], [0, 1, 0]])
    y_pred.requires_grad = True
    loss_fn = NllLoss()
    l = loss_fn(y_true, y_pred)
    print(l)
    l.backward()

