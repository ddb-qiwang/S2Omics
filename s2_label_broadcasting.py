# Autoencoder with batch norm layer
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCE_loss(nn.Module):
    def __init__(self, q=0.8, n=15):
        super(GCE_loss, self).__init__()
        self.q = q
        self.n = n

    def forward(self, outputs, targets):
        targets = torch.zeros(targets.size(0), self.n).cuda().scatter_(1, targets.view(-1, 1), 1)
        pred = F.softmax(outputs, dim=1)
        pred_y = torch.sum(targets * pred, dim=1)
        pred_y = torch.clamp(pred_y, 1e-4)
        final_loss = torch.mean((1.0 - pred_y ** self.q) / self.q, dim=0)
        return final_loss


class S2Omics_Predictor(nn.Module):

    def __init__(self, n_input, n_enc_1, n_enc_2, n_enc_3, n_z, n_cls_1, n_cls_out):
        super(S2Omics_Predictor, self).__init__()

        # encoder
        self.enc_layer_1 = nn.Sequential(
            nn.Linear(n_input, n_enc_1),
            #nn.BatchNorm1d(n_enc_1),
            nn.ReLU(inplace=True))
        self.enc_layer_2 = nn.Sequential(
            nn.Linear(n_enc_1, n_enc_2),
            #nn.BatchNorm1d(n_enc_2),
            nn.ReLU(inplace=True))
        self.enc_layer_3 = nn.Sequential(
            nn.Linear(n_enc_2, n_enc_3),
            #nn.BatchNorm1d(n_enc_3),
            nn.ReLU(inplace=True))
        self.z_layer = nn.Sequential(
            nn.Linear(n_enc_3, n_z),
            #nn.BatchNorm1d(n_z, affine=False)
        )

        # classifier
        self.cls_layer_1 = nn.Sequential(
            nn.Linear(n_z, n_cls_1),
            #nn.BatchNorm1d(n_cls_1),
            nn.ReLU(inplace=True))
        self.cls_out_layer = nn.Linear(n_cls_1, n_cls_out)

        # decoder
        self.dec_layer_1 = nn.Sequential(
            nn.Linear(n_z, n_enc_3),
            #nn.BatchNorm1d(n_enc_3),
            nn.ReLU(inplace=True))
        self.dec_layer_2 = nn.Sequential(
            nn.Linear(n_enc_3, n_enc_2),
            #nn.BatchNorm1d(n_enc_2),
            nn.ReLU(inplace=True))
        self.dec_layer_3 = nn.Sequential(
            nn.Linear(n_enc_2, n_enc_1),
            #nn.BatchNorm1d(n_enc_1),
            nn.ReLU(inplace=True))

        self.x_bar_layer = nn.Linear(n_enc_1, n_input)

    def forward(self, x):

        # encoder
        enc_h1 = self.enc_layer_1(x)
        enc_h2 = self.enc_layer_2(enc_h1)
        enc_h3 = self.enc_layer_3(enc_h2)

        z = self.z_layer(enc_h3)

        #classifier
        cls_h1 = self.cls_layer_1(z)
        cls_out = self.cls_out_layer(cls_h1)

        # decoder
        dec_h1 = self.dec_layer_1(z)
        dec_h2 = self.dec_layer_2(dec_h1)
        dec_h3 = self.dec_layer_3(dec_h2)
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z, cls_h1, cls_out