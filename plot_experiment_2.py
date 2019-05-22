if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    data = pd.read_csv('/Users/charlesdognin/Desktop/bagging_research_/old_res/rademacher_dist.csv')
    x, y = data['x'][7:], data['y'][7:]
    #
    # x1 = np.arange(2, 51)
    # x2= np.arange(1, 51)
    # y1 = [float(el.strip('\n')) for el in data]
    ###preparing the figure
    fig1 = plt.figure(1)
    ax=fig1.add_subplot(1, 1, 1)

    ###the three sets of data to plot
    ax.plot(x, y, linestyle='', marker='o', color='r', label=r'$MSE(\hat{\sigma},n)-MSE(\sigma,n)$')
    ax.plot(x, list(map(lambda x: (1 / x**2), x)), color='b', label=r'$\frac{-2\mu_4+3V^2}{n^2}$')

    ###beautification
    ax.legend(loc=0, fontsize=12)
    ax.set_ylabel(r'$MSE(\hat{\sigma},n)-MSE(\sigma,n)$')
    ax.set_xlabel("n")
    ax.grid()

    ###putting the plot
    plt.show()
    fig1.savefig('rademacher_distance.png')
