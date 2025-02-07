# Utility functions for PennyLane+PyTorch models
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for PL+Torch modelling
# Date: 2025

##### Initial settings

import sys
sys.path.append('.')
sys.path.append('..')
sys.path

import os
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
import numpy as numpy_np

# Pytorch imports
import torch
from torch import nn, tensor, optim


### QGraph wrapper with a static data structure and variant input and weights only
#   wires: circuit wires
#   n_data: number of data wires to be reserved
#   n_extra: number of extra wires to be used in training
#   n_layers: number of entangling layers to be produced
#   rot: rotation type, either 'Ry' or 'Rxyz'
#   scaler: scaler to be applied to the inputs
def qgraph_basis(wires, n_data, n_extra, n_layers=1, rot='Ry', scaler=np.pi):
    
    def _qgraph_circ(inputs, weights):
        # inputs: A single number, being a scaled (down) vertex id
        #         Note that when a NN generates inputs, its results will be in range [-1..1]
        # output: probability distribution of applying the circuit shot number of times
        nonlocal wires, n_data, n_extra, n_layers, rot, scaler
        
        n_learn = n_data + n_extra
        data_wires = wires[0:n_data]
        learn_wires = wires[0:n_learn]
        scaled_inputs = torch.mul(inputs, scaler)
        qml.AngleEmbedding(scaled_inputs, wires=data_wires)
        
        if rot == 'Ry':
            qml.BasicEntanglerLayers(weights, rotation=qml.RY, wires=learn_wires)
        elif rot == 'Rxyz':
            qml.StronglyEntanglingLayers(weights, wires=learn_wires)
        # return [qml.expval(qml.PauliZ(wires=w)) for w in data_wires]
        return qml.probs(wires=data_wires)
    return _qgraph_circ

def qgraph_basis_shape(n_data, n_extra, n_layers=1, rot='Ry'):
    n_wires = n_data + n_extra
    if rot == 'Ry':
        shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_wires)
    elif  rot == 'Rxyz':
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
    else:
        shape = (0)
    return shape



##### Hybrid QGraph
#
#   Graph properties:
#      wires: circuit wires
#      n_data: number of data wires to be reserved
#      n_extra: number of extra wires to be used in training
#      n_layers: number of entangling layers to be produced
#      rot: rotation type, either 'Ry' or 'Rxyz'
#      scaler: scaler to be applied to all inputs
#      mode: QGraph type, i.e. 'classic', 'quantum' or 'hybrid'
#
#   Circuit functionality (input -> output):
#      input - vertex id (log2N qubits)
#      output - edge weight distribution / probability distribution of target vertex selection (N results)
class qgraph_model(nn.Module):

    def __init__(self, sim, n_data, n_extra, n_layers=1, rot='Ry', shots=0, scaler=np.pi, seed=0, ws=0.1):
        super(qgraph_model, self).__init__()

        self.qg_mode = 'q'
        self.sim = sim
        self.n_data = n_data
        self.n_extra = n_extra
        self.n_wires = n_data+n_extra
        self.wires = list(range(self.n_wires))
        self.n_layers = n_layers
        self.rot = rot
        self.shots = shots
        self.scaler = scaler
        self.seed = rand_seed() if seed == 0 else seed
        self.wscaler = ws
        self.mode('quantum')

        # Create the model
        def_qlayer = self.qlayer()
        layers = self.layers(def_qlayer)        
        self.model = nn.Sequential(*layers)  

    ### Define an optional (default) quantum layer
    def qlayer(self):
        # Define QGraph circuit weight shapes
        tensor_shape = qgraph_basis_shape(self.n_data, self.n_extra, n_layers=self.n_layers, rot=self.rot)
        weights_shapes = {"weights": tensor_shape}
        torch.manual_seed(self.seed)
        weights = torch.rand(tensor_shape, requires_grad=True) * self.wscaler
        init_method = {"weights": weights} # torch.nn.init.normal_ # torch.nn.init.uniform_

        # Define QGraph circuit and its layer
        qgraph = qgraph_basis(self.wires, self.n_data, self.n_extra, n_layers=self.n_layers, rot=self.rot, scaler=self.scaler)
        if self.shots == 0:
            dev = qml.device(self.sim, wires=self.n_wires)
        else:
            dev = qml.device(self.sim, wires=self.n_wires, shots=self.shots)
        qgraph_node = qml.QNode(qgraph, dev, interface='torch') #, diff_method='spsa') #, level='gradient')
        qlayer = qml.qnn.TorchLayer(qgraph_node, weight_shapes=weights_shapes, init_method=init_method)
        return qlayer
        
    ### Define all qgraph layers and return their list
    def layers(self, qlayer):
        ### Default only a quantum layer
        return [qlayer]

    ### Set or return the qgraph mode (any string)
    def mode(self, vmode=None):
        if vmode is not None:
            self.qg_mode = vmode
        return self.qg_mode

    ### Apply the model to data
    def forward(self, x):
        x = self.model(x)
        return x

#####
##### Several subclassess to illustrate how to vary the model
##### Notes:
#####    Models are named depending on their type, which is reflected in their starting letter, e.g.
#####       c - Purely classical models (no quantum layer)
#####       q - Purely quantum models (no classical layers)
#####       h - Hybrid models (including both classical and hybrid layers in some configuration)
#####    It is important to match model complexity with qgraph size
#####       this is why 8 vertex models are called xxx_8, 16 vertex models xxx_16, etc.
#####

class qgraph_classic_8(qgraph_model):
    def __init__(self, sim, n_data, n_extra, n_layers=1, rot='Ry', shots=0, scaler=np.pi, seed=0, ws=0.1):
        super(qgraph_classic_8, self).__init__(sim, n_data, n_extra, n_layers, rot, shots, scaler, seed, ws)
        self.mode('c8')

    def layers(self, qlayer):
        # Define all layers including QNN layer
        clayer_0 = torch.nn.Linear(self.n_data, self.n_data)
        clayer_1 = torch.nn.ReLU()
        clayer_2 = torch.nn.Linear(self.n_data, 20)
        clayer_3 = torch.nn.ReLU()
        clayer_4 = torch.nn.Linear(20, self.n_data)
        clayer_5 = torch.nn.Linear(self.n_data, 2**self.n_data)
        
        layers = [clayer_0, clayer_1, clayer_2, clayer_3, clayer_4, clayer_5]
        return layers

class qgraph_classic_8_softmax(qgraph_model):
    def __init__(self, sim, n_data, n_extra, n_layers=1, rot='Ry', shots=0, scaler=np.pi, seed=0, ws=0.1):
        super(qgraph_classic_8_softmax, self).__init__(sim, n_data, n_extra, n_layers, rot, shots, scaler, seed, ws)
        self.mode('cs8')

    def layers(self, qlayer):
        # Define all layers including QNN layer
        clayer_0 = torch.nn.Linear(self.n_data, self.n_data)
        clayer_1 = torch.nn.ReLU()
        clayer_2 = torch.nn.Linear(self.n_data, 20)
        clayer_3 = torch.nn.ReLU()
        clayer_4 = torch.nn.Linear(20, self.n_data)
        clayer_5 = torch.nn.Linear(self.n_data, 2**self.n_data)
        
        layers = [clayer_0, clayer_1, clayer_2, clayer_3, clayer_4, clayer_5]
        return layers

    def forward(self, x):
        x = self.model(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

class qgraph_quantum_8(qgraph_model):
    def __init__(self, sim, n_data, n_extra, n_layers=1, rot='Ry', shots=0, scaler=np.pi, seed=0, ws=0.1):
        super(qgraph_quantum_8, self).__init__(sim, n_data, n_extra, n_layers, rot, shots, scaler, seed, ws)
        self.mode('q8')

    def layers(self, qlayer):        
        layers = [qlayer]
        return layers

class qgraph_hybrid_8(qgraph_model):
    def __init__(self, sim, n_data, n_extra, n_layers=1, rot='Ry', shots=0, scaler=np.pi, seed=0, ws=0.1):
        super(qgraph_hybrid_8, self).__init__(sim, n_data, n_extra, n_layers, rot, shots, scaler, seed, ws)
        self.mode('hs8')

    def layers(self, qlayer):
        # Define all layers including QNN layer
        clayer_0 = torch.nn.Linear(self.n_data, self.n_data)
        clayer_1 = torch.nn.ReLU()
        clayer_2 = torch.nn.Linear(self.n_data, 20)
        clayer_3 = torch.nn.ReLU()
        clayer_4 = torch.nn.Linear(20, self.n_data)
        
        layers = [clayer_0, clayer_1, clayer_2, clayer_3, clayer_4, qlayer]
        return layers

class qgraph_classic_16(qgraph_model):
    def __init__(self, sim, n_data, n_extra, n_layers=1, rot='Ry', shots=0, scaler=np.pi, seed=0, ws=0.1):
        super(qgraph_classic_16, self).__init__(sim, n_data, n_extra, n_layers, rot, shots, scaler, seed, ws)
        self.mode('c16')

    def layers(self, qlayer):
        # Define all layers including QNN layer
        slayer_0 = torch.nn.Linear(self.n_data, 50)
        slayer_1 = torch.nn.ReLU()
        slayer_2 = torch.nn.Linear(50, 100)
        slayer_3 = torch.nn.ReLU()
        slayer_4 = torch.nn.Linear(100, 50)
        slayer_5 = torch.nn.ReLU()
        slayer_6 = torch.nn.Linear(50, self.n_data)
        slayer_7 = torch.nn.Linear(50, 2**self.n_data)
        
        layers = [slayer_0, slayer_1, slayer_2, slayer_3, slayer_4, slayer_5, slayer_7]
        return layers

class qgraph_classic_16_softmax(qgraph_model):
    def __init__(self, sim, n_data, n_extra, n_layers=1, rot='Ry', shots=0, scaler=np.pi, seed=0, ws=0.1):
        super(qgraph_classic_16_softmax, self).__init__(sim, n_data, n_extra, n_layers, rot, shots, scaler, seed, ws)
        self.mode('cs16')

    def layers(self, qlayer):
        # Define all layers including QNN layer
        slayer_0 = torch.nn.Linear(self.n_data, 50)
        slayer_1 = torch.nn.ReLU()
        slayer_2 = torch.nn.Linear(50, 100)
        slayer_3 = torch.nn.ReLU()
        slayer_4 = torch.nn.Linear(100, 50)
        slayer_5 = torch.nn.ReLU()
        slayer_6 = torch.nn.Linear(50, self.n_data)
        slayer_7 = torch.nn.Linear(50, 2**self.n_data)
        
        layers = [slayer_0, slayer_1, slayer_2, slayer_3, slayer_4, slayer_5, slayer_7]
        return layers

    def forward(self, x):
        x = self.model(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

class qgraph_quantum_16(qgraph_model):
    def __init__(self, sim, n_data, n_extra, n_layers=1, rot='Ry', shots=0, scaler=np.pi, seed=0, ws=0.1):
        super(qgraph_quantum_16, self).__init__(sim, n_data, n_extra, n_layers, rot, shots, scaler, seed, ws)
        self.mode('q16')

    def layers(self, qlayer):        
        layers = [qlayer]
        return layers

class qgraph_hybrid_16(qgraph_model):
    def __init__(self, sim, n_data, n_extra, n_layers=1, rot='Ry', shots=0, scaler=np.pi, seed=0, ws=0.1):
        super(qgraph_hybrid_16, self).__init__(sim, n_data, n_extra, n_layers, rot, shots, scaler, seed, ws)
        self.mode('h16')

    def layers(self, qlayer):
        # Define all layers including QNN layer
        slayer_0 = torch.nn.Linear(self.n_data, 50)
        slayer_1 = torch.nn.ReLU()
        slayer_2 = torch.nn.Linear(50, 100)
        slayer_3 = torch.nn.ReLU()
        slayer_4 = torch.nn.Linear(100, 50)
        slayer_5 = torch.nn.ReLU()
        slayer_6 = torch.nn.Linear(50, self.n_data)
        
        layers = [slayer_0, slayer_1, slayer_2, slayer_3, slayer_4, slayer_5, slayer_6, qlayer]
        return layers
