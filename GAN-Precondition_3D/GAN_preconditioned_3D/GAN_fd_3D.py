"""
frequency-domain reverse time migration
use OFD methods for forward modeling
GAN-based preconditioners for RTM
illumination-compensated cross-correlation imaging conditions
the whole PML absorbing condition
final revision: 2024.8.28
"""
#Taking the Sunken model as an example
import torch
import time
from ofd_matrix_4_3D import matrix3fd4, source_3ofd
import numpy as np
from bicgstab_3D import bicgstab_model
from generator_3D import UNet
##################################################
################## Basic parameters ##############
##################################################
nx = 66
ny = 66
nz = 66
# spatial step
h = 0.035
# dominant frequency of source
f = 20
f0 = 10
# numbers of PML layers
delta = 10
#c_vec is the wave velocity
cc = 4 * np.ones((nz - 2, nx - 2, ny - 2), dtype=float)
N = (nx-2)*(ny-2)*(nz-2)
c_vec = cc.reshape(N, 1)

#s_loc = int((nx-2)*(ny-2)*(nz-2)//2+(nx-2)*(ny-2)//2);
ix = (nx - 2) // 2
iy = (ny - 2) // 2
iz = 8
s_loc = iz * (nx - 2) * (ny - 2) + iy * (nx - 2) + ix
bzeros = np.zeros((nx-2, ny-2, nz-2), dtype=complex)
bb = source_3ofd(f, f0, N, h, c_vec, s_loc).todense()
bzeros[(nx-2)//2, (ny-2)//2, 8] = bb[bb.nonzero()].real + 1j * bb[bb.nonzero()].imag

# A = matrix_ofd4(nx, nz, h, f, delta, c_vec)
A = matrix3fd4(nx, ny, nz, h, h, h, f, delta, c_vec)
b = bzeros.reshape(N, 1)
Gf = np.min(cc)/(2.5*f0*h)
print(Gf)

##### 加载训练好的网络参数####
import torch
from collections import OrderedDict
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 创建模型
model = UNet()
# 读取权重
state_dict = torch.load('netG_weights.pth', map_location=device)
# 如果 key 里有 "module."，就去掉
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")  # 去掉多余的前缀
    new_state_dict[name] = v
# 加载处理后的权重
model.load_state_dict(new_state_dict)
# 移动到设备
model.to(device)

#######  solve Ax = b system #######
# ### Preconditioning GAN ###
start_time = time.time()
_, error_nn,it_nn, resdual_nn = bicgstab_model(A, b, model)
end_time_nn = time.time() - start_time
print('vinnbicgstab:', 'error:', error_nn, 'iter:', it_nn, 'time:', end_time_nn)

results = {
    'error_nn': error_nn,
    'it_nn': it_nn,
    'resdual_nn': resdual_nn
}
# 写入字典到文件
import json
with open('3DGANjunyun_results.json', 'w') as f:
    json.dump(results, f)