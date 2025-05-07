
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from matplotlib.colors import to_rgba
import seaborn as sns
from tqdm.notebook import tqdm
from tqdm import trange
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch._VF as _VF
from torch.utils.tensorboard import SummaryWriter
import json
import random
import sys
import argparse

simglucose_test_results_path = '/content/drive/MyDrive/ColabNotebooks/Medical_case/test_results'
simglucose_patient_data_path = '/content/drive/MyDrive/ColabNotebooks/Medical_case/patient_data'
# Simulation parameters:
# Start date: 2022-06-30
# Input simulation time (hr): 720h
# Random Scnenario
# Input simulation start time (hr): 0:00
# Select random seed for random scenario: 15
# Select the CGM sensor: Dexcom
# Select Random Seed for Sensor Noise: 10
# Select the insulin pump: Insulet
# Select controller: Basal-Bolus Controller

from simglucose.simulation.user_interface import simulate
simulate()
