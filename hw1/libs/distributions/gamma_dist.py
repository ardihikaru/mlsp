import scipy.stats as stats
import seaborn as sns

class GammaDist():
    def __init__(self, **kwargs):
        self.__data_initialization(**kwargs)
        self.__generate_dataset()

    def __data_initialization(self, **kwargs):
        self.loc = kwargs["loc"]
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.size = kwargs["size"]
        self.tetha = 1 / self.beta
        # self.st_range = kwargs["st_range"]

    def get_size(self):
        return self.size

    def __generate_dataset(self):
        self.rand_gamma = stats.gamma.rvs(self.alpha, scale=self.tetha, size=self.size)
        # self.rand_exp = np.random.exponential(self.beta, self.size)

    def __update_alpha(self, new_alpha):
        self.alpha = new_alpha

    def regenerate_dataset(self, new_alpha=None):
        if new_alpha is not None:
            self.__update_alpha(new_alpha)
        self.__generate_dataset()

    def get_dataset(self):
        return self.rand_gamma

    def get_alpha(self):
        return self.alpha

    def get_beta(self):
        return self.beta

    def plot_dataset(self):
        ax = sns.distplot(self.rand_gamma,
                          kde=True,
                          bins=self.size, # number of sliced data (number of x in plot)
                          color='black',
                          # color='skyblue',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Gamma Distribution', ylabel='Frequency')

