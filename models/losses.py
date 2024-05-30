import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, BINARY_MODE


class Phase1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss(BINARY_MODE, from_logits=True)

    def forward(self, out, conf, det, mask, label):
        return self.dice_loss(out, mask)


class Phase2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.confidence_loss = nn.MSELoss()
        self.detection_loss = nn.BCEWithLogitsLoss()

    def forward(self, out, conf, det, mask, label):
        out = F.sigmoid(out)
        t_map = out * mask + (1 - out) * (1 - mask)
        return self.confidence_loss(conf, t_map) + 0.5 * self.detection_loss(det, label)
