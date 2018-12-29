import numpy as np

Nomal_WT = np.loadtxt("Normalized_WT_Lite_Train.dat")
np.mat(Nomal_WT)
Dmax = Nomal_WT.max()
Dmin = Nomal_WT.min()
Nomal_WT = (Nomal_WT - Dmin) / (Dmax - Dmin)
print(Dmin)
print(Dmax)
print(Nomal_WT)
filename = 'Normalized_Wavelet.dat'
with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    for i in range(0, 27806):
        for k in range(0, 127):
            f.write(str(Nomal_WT[i][k]))
            f.write(' ')
        f.write('\n')
