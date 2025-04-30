from torchmetrics import Metric
import torch
import torch.nn.functional as F

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes=200, eps=1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.add_state("tp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("pp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("ap", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds, target):
        if preds.dim() > 1 and preds.size(1) > 1:
            preds = preds.argmax(dim=1)
        else:
            preds = (preds > 0.5).long().squeeze()

        preds = torch.clamp(preds, 0, self.num_classes - 1)
        target = torch.clamp(target, 0, self.num_classes - 1)
        preds_oh  = F.one_hot(preds,  num_classes=self.num_classes)
        target_oh = F.one_hot(target, num_classes=self.num_classes)

        tp_vec = (preds_oh & target_oh).sum(dim=0)
        pp_vec = preds_oh.sum(dim=0)
        ap_vec = target_oh.sum(dim=0)

        self.tp += tp_vec.sum()
        self.pp += pp_vec.sum()
        self.ap += ap_vec.sum()

    def compute(self):
        tp = self.tp.float()
        pp = self.pp.float()
        ap = self.ap.float()
        return 2 * tp / (pp + ap + self.eps)


class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds_classes = preds.argmax(dim=1)
        # print('preds_classes',preds_classes.shape,preds_classes)
        # print('target',target.shape,target)

        # [TODO] check if preds and target have equal shape
        assert preds_classes.shape == target.shape, f'preds_classes.shape {preds_classes.shape} != target.shape {target.shape}'

        # [TODO] Cound the number of correct prediction
        correct = (preds_classes == target).sum()
        # print((preds_classes == target).float().mean(), (preds_classes == target).float())
        # print('sum',correct)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
