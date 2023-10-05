from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import math
"""
For RNN Training:
- Stateful and Stateless are two different cases
- we choose stateless cases here.
https://intellipaat.com/community/14887/shuffling-training-data-with-lstm-rnn
"""

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        pass

class VRNNCell_V0(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim):
        super(VRNNCell_V0, self).__init__()
        # init params
        self.x_dim, self.z_dim, self.h_dim = x_dim, z_dim, h_dim
        # Feature Extractors
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim),nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim),nn.ReLU())

        # Note: only the output layers of std have activiation softplus
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(2*h_dim,h_dim),nn.ReLU(),
            nn.Linear(h_dim, h_dim),nn.ReLU()
            )
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(nn.Linear(h_dim, z_dim),nn.Softplus())

        #prior
        self.prior = nn.Sequential(nn.Linear(h_dim, h_dim),nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim),nn.Softplus())

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2*h_dim,h_dim),nn.ReLU(),
            nn.Linear(h_dim, h_dim),nn.ReLU()
            )
        # self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Linear(h_dim, x_dim)
        # self.dec_std = nn.Sequential(nn.Linear(h_dim, x_dim),nn.Softplus())
        
        self.rnn = nn.GRUCell(input_size=2*h_dim, hidden_size=h_dim)
        self._recon_loss = nn.MSELoss()
    def _reparameterized_sample(self, mean, std):
        epsilon = torch.rand_like(mean, device=mean.device)
        return mean + epsilon*std

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return  0.5 * torch.mean(kld_element)


    def forward(self, x):
        """
        :param x(list of namedtuples): shape of [time, batch, features]
        """
        # all_enc_mean, all_enc_std = [], []
        # all_dec_mean, all_dec_std = [], []
        kld_loss, recon_loss = 0, 0

        # stateless between different sequences, thus initialize h for each sequence
        h = torch.zeros([x[0].size()[0], self.h_dim], device=x[0].device)
        for t in range(len(x)):
            # Inference Model
            temp_x = x[t]
            x_t = self.phi_x(temp_x)
            enc_t = self.encoder(torch.cat([x_t,h],dim=1))
            enc_mean_t, enc_std_t = self.enc_mean(enc_t), self.enc_std(enc_t)
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t = self.phi_z(z_t)
            # Recurence
            h = self.rnn(torch.cat([x_t,z_t],dim=1), h)


            if self.training:
                # Reconsruct \hat{x}
                dec_t = self.decoder(torch.cat([x_t,z_t],dim=1))
                dec_mean_t = self.dec_mean(dec_t)
                # dec_std_t = self.dec_std(dec_t)
                # Estimate Prior distribution of z
                prior_t = self.prior(h)
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)

                # all_enc_mean.append(enc_std_t)
                # all_enc_std.append(enc_mean_t)
                # all_dec_mean.append(dec_mean_t)
                # all_dec_std.append(dec_std_t)
                kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
                recon_loss += self._recon_loss(temp_x, dec_mean_t)

         # model.eval() will set this flag to false
        if self.training:
            return recon_loss, kld_loss
        else:
            # return the h of the last time step as belief state
            return h



class VRNNCell_V1(nn.Module):
    def __init__(self):
        pass



if __name__=="__main__":
    model = VRNNCell_V0(x_dim=5,z_dim=10,h_dim=15)
    model.train()
    # Data format x:(time, batch, features)
    sample_x = torch.rand((10,15,5))
    loss = model(sample_x)
    print(loss)
    model.eval()
    sample_x = torch.ones((10,15,5))
    h = model(sample_x)

    sample_x = [torch.ones(15,5) for i in range(10)]
    h = model(sample_x)
    h = model(sample_x)
    print(h.size())
