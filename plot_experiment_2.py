if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    data = open('/Users/charlesdognin/Desktop/bagging_research_/results_mse_cali|0.5|400|2|1000.txt', 'r').readlines()

    x = 0
    y1 = 0
    ###preparing the figure
    fig1 = plt.figure(1)
    ax=fig1.add_subplot(1,1,1)

    ###the three sets of data to plot
    ax.plot(x[6:], y1[6:], linestyle='', marker='o', color='r', label=r'$MSE(\hat{\sigma},n)-MSE(\sigma,n)$')
    ax.plot(x, list(map(lambda x: (1 / x**2), x)), color='b', label=r'$\frac{-2\mu_4+3V^2}{n^2}$')

    ###beautification
    ax.legend(loc=0, fontsize=12)
    ax.set_ylabel(r'$MSE(\hat{\sigma},n)-MSE(\sigma,n)$')
    ax.set_xlabel("n")
    ax.grid()

    ###putting the plot
    plt.show()
