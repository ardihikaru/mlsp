#%%

# Source: https://stackoverflow.com/questions/2896179/fitting-a-gamma-distribution-with-python-scipy
import scipy.stats as stats
import seaborn as sns
alpha = 1
loc = 0
beta = 100
tetha = 1/beta
size=20
# data = stats.gamma.rvs(alpha, loc=loc, scale=beta, size=20)
data = stats.gamma.rvs(alpha, scale=tetha, size=size)
print(data)

ax = sns.distplot(data,
                  kde=True,
                  bins=20,
                  color='black',
                  # color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Gamma Distribution', ylabel='Frequency')