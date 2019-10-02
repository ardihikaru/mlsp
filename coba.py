import numpy as np
import matplotlib.pyplot as plt

# target = 100
size = 20
# alpha = 1
# beta = 1.0/target
lam = 0.2
# target = 0.5
# target = 2.5*lam #
beta = 1 / lam # beta = 1 / lambda


Y = np.random.exponential(beta, size)

# plt.plot(x, Y, 'b-')
# plt.plot(x[:size], Y, 'r.')
# # plt.plot(x[:size], simulated_data, 'r.')
# plt.show()

# bin = jumlah patahan
# alpha = bar's transparancy; value = 0-1 (decimal)
plt.hist(Y, density=True, bins=size*2, lw=100, alpha=.9)
# # plt.hist(Y, density=True, bins=4, lw=0, alpha=.8)
# # plt.hist(Y, density=False, bins=200,lw=0,alpha=.8)
# plt.plot([0, max(Y)], [target, target], 'r--')
# # plt.ylim(0,target*1.1)
plt.show()

