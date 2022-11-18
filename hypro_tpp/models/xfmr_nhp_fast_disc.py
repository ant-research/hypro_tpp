import torch
from torch import nn
from .xfmr_nhp_fast import XFMRNHPFast


class XFMRNHPFastDisc(XFMRNHPFast):
    def __init__(self, model_config):
        super(XFMRNHPFastDisc, self).__init__(model_config)

        # prediction for discriminator
        self.discriminator_prediction_layer = torch.nn.Sequential(
            nn.Linear(self.d_model * self.n_head, self.d_model),
            nn.ReLU(),
            nn.Dropout(model_config['dropout']),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(model_config['dropout']),
            nn.Linear(self.d_model, 1),
        )

        self.discriminator_loss = nn.CrossEntropyLoss(reduction='mean')

    def predict_as_discriminator(self, logits):
        logits = self.discriminator_prediction_layer(logits)
        return logits
