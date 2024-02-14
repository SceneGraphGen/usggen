# pytorch
import torch
from torch import Tensor
from torch.utils.data import sampler, DataLoader, Dataset
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR, ExponentialLR, StepLR
#from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
#import pytorch_warmup as warmup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# utilities
import time
import math
from math import log, sqrt
import numpy as np
import random
from random import shuffle
import h5py
import json
import pickle
import os
import itertools
import collections
#import pandas as pd
#import seaborn as sn
from collections import Counter
from matplotlib import pyplot as plt
import tempfile
from imageio import imread, imwrite

# ML lib
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
#import gensim
# graph
import networkx as nx
from graphviz import Digraph
#from grakel import Graph, kernels

EDGE_INFO = True

# # for R packages
# import rpy2.robjects.numpy2ri
# from rpy2.robjects.packages import importr
# rpy2.robjects.numpy2ri.activate()
# stats = importr('stats')
