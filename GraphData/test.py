import pandas as pd
import scipy.io as scio
data_path="netS_ppmi_4_1.mat"

#Method 1

data = scio.loadmat(data_path)
print(data['pp_netS'])