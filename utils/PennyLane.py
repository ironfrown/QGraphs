# Utility functions for PennyLane programming
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for PL development
# Date: 2024

##### Initial settings

import sys
sys.path.append('.')
sys.path.append('..')
sys.path

import os
import numpy as numpy_np
import pylab
import math
import json

import networkx as nx
from networkx.readwrite import json_graph

import matplotlib.pyplot as plt
from matplotlib import set_loglevel
set_loglevel("error")

from IPython.display import clear_output

import pennylane as qml
from pennylane import numpy as np

# Pytorch imports
import torch
from torch import nn, tensor, optim


##### Small functions

### Bit to list translation for data entry and state interpretation
#   Note: PennyLane interprets qubits state in reverse order than Qiskit
#         These functions are a copy of functions in Circuits.py

### Transform int number to a list of bits, bit 0 comes first
def bin_int_to_list(a, n_bits):
    a = int(a)
    a_list = [int(i) for i in f'{a:0{n_bits}b}']
    # a_list.reverse()
    return numpy_np.array(a_list)

### Transform a list of bits to an int number, bit 0 comes first
def bin_list_to_int(bin_list):
    b = list(bin_list)
    # b.reverse()
    return int("".join(map(str, b)), base=2)

### Converts a list/array of numbers to a list of their binary representations as a list
def nums_to_bin_tensor(num_list, n_data, torch_device='cpu'):
    bin_list_list = np.array([bin_int_to_list(n, n_data) for n in num_list])
    tens_list = torch.tensor(bin_list_list, dtype=torch.double)
    return tens_list.to(torch_device)

### Counts the number of purely pytorch model parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

### Gets all pytorch parameters
def get_param_vals(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)
    params = params.cpu().detach().flatten() # [0]
    return params.numpy()


##### Jensen-Shannon Divergence

# Refs: 
# - Amin Jun + Renly Hou: https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/9
# - johnball: https://stats.stackexchange.com/questions/303735/jensen-shannon-divergence-for-multiple-probability-distributions

### KL and JSD divergence metrics
#   All metrics arguments follow the order according to the PyTorch convention
#   - P: obtained output
#   - W: expected targets
class div_metrics(nn.Module):
    def __init__(self):
        super(div_metrics, self).__init__()

    # KL cost function, assumes equal shapes of W and P
    # Removes all (w, p) pairs which have zeros
    # - P: obtained output
    # - W: expected targets
    def kl(self, P: torch.tensor, W: torch.tensor):
        vn = W.shape[0]
        W, P = torch.flatten(W), torch.flatten(P)
        kl = sum([w * math.log2(w / p) for (w, p) in zip(W, P) if (w > 0) and (p > 0)]) / vn
        return kl

    # JSD cost function, assumes equal shapes of W and P
    # Removes all (w, p) pairs which have zeros
    # - P: obtained output
    # - W: expected targets
    def jsd(self, P: torch.tensor, W: torch.tensor):
        M = 0.5 * (W + P)
        jsd = 0.5 * (self.kl(M, W) + self.kl(M, P))
        return jsd

    # JSD loss function (so a single vertex data)
    def jsd_single_loss(self, outputs: torch.tensor, targets: torch.tensor):
        self.jsd(torch.tensor([outputs]), torch.tensor([targets]))


##### PL probability distributions

### Plot probability distribution
#   probs: list or tensor
#   thres: all probs less that threshold will not be plotted
def plot_hist(probs, scale=None, figsize=(8, 6), dpi=72, th=0, title='Measurement Outcomes'):

    # Prepare data
    n_probs = len(probs)
    n_digits = len(bin_int_to_list(n_probs, 1)) # 1 means as many digits as required
    labels = [f'{n:0{n_digits}b}' for n in np.arange(n_probs)]

    # Filter out the prob values below threshold
    pairs = [(p, l) for (p, l) in zip(probs, labels) if p >= th]
    probs = [p for (p, l) in pairs]
    labels = [l for (p, l) in pairs]

    # Plot the results
    fig, ax=plt.subplots(figsize=figsize, dpi=dpi)
    ax.bar(labels, probs)
    ax.set_title(title)
    plt.xlabel('Results')
    plt.ylabel('Probability')
    plt.xticks(rotation=60)
    if scale is not None:
        dpi = fig.get_dpi()
        fig.set_dpi(dpi*scale)
    plt.show()

### Plot probability distribution
#   probs: list or tensor
#   thres: all probs less that threshold will not be plotted
def plot_compare_hist(probs_1, probs_2, scale=None, figsize=(8, 6), dpi=72, th=0, 
                      title_1='Measurement Outcomes 1', title_2='Measurement Outcomes 2',
                      xlabel_1='Results', xlabel_2='Results',
                      ylabel_1='Probability', ylabel_2='Probability'):

    # Prepare data
    n_probs_1 = len(probs_1)
    n_digits_1 = len(bin_int_to_list(n_probs_1, 1)) # 1 means as many digits as required
    labels_1 = [f'{n:0{n_digits_1}b}' for n in np.arange(n_probs_1)]
    n_probs_2 = len(probs_2)
    n_digits_2 = len(bin_int_to_list(n_probs_2, 1)) # 1 means as many digits as required
    labels_2 = [f'{n:0{n_digits_2}b}' for n in np.arange(n_probs_2)]

    # Filter out the prob values below threshold
    pairs_1 = [(p, l) for (p, l) in zip(probs_1, labels_1) if p >= th]
    probs_1 = [p for (p, l) in pairs_1]
    labels_1 = [l for (p, l) in pairs_1]
    pairs_2 = [(p, l) for (p, l) in zip(probs_2, labels_2) if p >= th]
    probs_2 = [p for (p, l) in pairs_2]
    labels_2 = [l for (p, l) in pairs_2]

    # Plot the results
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    axs[0].bar(labels_1, probs_1)
    axs[0].set_title(title_1)
    axs[0].set_xlabel(xlabel_1)
    axs[0].set_ylabel(ylabel_1)
    axs[0].tick_params(labelrotation=60)
    axs[1].bar(labels_2, probs_2)
    axs[1].set_title(title_2)
    axs[1].set_xlabel(xlabel_2)
    axs[1].set_ylabel(ylabel_2)
    axs[1].tick_params(labelrotation=60)

    if scale is not None:
        dpi = fig.get_dpi()
        fig.set_dpi(dpi*scale)
    plt.show()


