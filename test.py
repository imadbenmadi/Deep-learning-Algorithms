import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

# For reproducibility : to make sure your results are the same every time you run the code.
# ensures that NumPy operations behave predictably.
# ensures that PyTorch operations are also deterministic (as much as possible).
np.random.seed(42)
torch.manual_seed(42)

# Set up matplotlib for better visualizations
# plt.style.use('seaborn-whitegrid')