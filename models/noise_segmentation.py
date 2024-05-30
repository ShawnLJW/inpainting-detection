import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_extraction import SRMFilter
from .cmx.encoders.dual_segformer import mit_b2
from .cmx.decoders.MLPDecoder import DecoderHead


class NoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm_conv = SRMFilter()
        self.encoder = mit_b2()
        self.decoder = DecoderHead(num_classes=1, embed_dim=512)
        self.confidence_decoder = DecoderHead(num_classes=1, embed_dim=512)
        self.clf_head = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 1),
        )

    def forward(self, rgb):
        srm = self.srm_conv(rgb)
        hidden_states = self.encoder(rgb, srm)
        
        out = self.decoder(hidden_states)
        out = F.interpolate(
            input=out,
            size=(out.shape[-2] * 4, out.shape[-1] * 4),
            mode="bilinear",
        )

        conf = self.confidence_decoder(hidden_states)
        conf = F.interpolate(
            input=conf,
            size=(conf.shape[-2] * 4, conf.shape[-1] * 4),
            mode="bilinear",
        )

        det = torch.stack(self.get_stats(out, conf) + self.get_stats(conf, torch.zeros_like(conf)), dim=1)
        det = self.clf_head(det)[:, 0]

        return out, conf, det
    
    def get_stats(self, x, log_w):
        b = x.size(0)
        x = x.view(b, -1)
        log_w = log_w.view(b, -1)
        log_w = F.log_softmax(log_w, dim=-1)
        w = torch.exp(log_w)
        x_avg = torch.sum(w * x, dim=-1)
        x_msq = torch.sum(w * x * x, dim=-1)
        x_max = torch.logsumexp(log_w + x, dim=-1)
        x_min = -torch.logsumexp(log_w - x, dim=-1)
        return [x_avg, x_msq, x_max, x_min]

    def init_weights(self, mit_path=None):
        if mit_path:
            assert os.path.isfile(mit_path), "Invalid path to MiT-B2 weights"
            state_dict = torch.load(mit_path, map_location=torch.device("cpu"))
            self.encoder.load_state_dict(state_dict, strict=False)
            