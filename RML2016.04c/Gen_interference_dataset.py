# 作者：ruby
# 开发时间：2023/1/2 19:29
import numpy as np
import torch,pickle
import matplotlib.pyplot as plt
from model import ConvNet128
from data.data_processing import load_data, dataloader, get_distubution_power
from data.data_processing import get_signal_noise_power, find_snr
from attack_interference import *
from mlxtend.plotting import plot_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_dir = 'F:/dataset/RML/2016.04C.multisnr.pkl'
all_data, divide_data = load_data(file_dir)
[x_train, x_test, y_train, y_test, _, lbl_test] = divide_data
snrs_data= list(map(lambda x: int(x[1]), all_data[2]))
mods_data = list(map(lambda x: x[0], all_data[2]))
snrs_data, mods_data = np.array(snrs_data),np.array(mods_data)
print('mods_data',snrs_data.shape,mods_data.shape)

trainloader = dataloader(x_train, y_train)


model = ConvNet128(len(all_data[-1])).to(device=device)
model.load_state_dict(torch.load('./data/saved_Convnetmodel(rml2016c).pth', map_location=device))
model.eval()

v = generate(torch.Tensor(x_train), torch.Tensor(y_train),
                 trainloader, model, pca=True)
dict = {}
for snr in all_data[3]:
    snr_index = np.where(snrs_data==snr)[0]
    data_snr= all_data[0][snr_index]
    mods_snr = mods_data[snr_index]
    s_power, n_power = get_signal_noise_power(data_snr, snr)
    for i in range(len(all_data[-1])):
        mod = all_data[-1][i]
        mod_index = np.where(mods_snr == mod)[0]
        data_mod = data_snr[mod_index]
        for inr in range(-10, 12, 2):
            v_am = get_distubution_power(n_power, inr) ** 0.5
            v_vector = v * v_am
            data_v = np.squeeze(data_mod+v_vector)
            dict[(mod,snr,inr)]=data_v

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
savepath ='F:\dataset\small\RML2016.10c_int'
save_obj(dict, savepath)




