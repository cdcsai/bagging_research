if __name__ == '__main__':
    import pandas as pd
    csv = False

    data = open('/home/verisk/Desktop/bagging_research/models/res_special_rad_100000.txt', 'r').readlines()
    k = [float(el.split('|')[0]) for el in data]
    mse_var = [float(el.split('|')[1]) for el in data]
    mse_var_bag = [float(el.split('|')[2]) for el in data]

    import matplotlib.pyplot as plt
    import numpy as np

    ###preparing the figure
    fig1 = plt.figure(1)
    ax=fig1.add_subplot(1,1,1)

    ###the three sets of data to plot

    ax.plot(k, mse_var, linestyle='', marker='o', color='r', label=r'MSE without Bagging')
    ax.plot(k, mse_var_bag, linestyle='', marker='o', color='b', label=r'MSE with Bagging')
    plt.axvline(x=3 / 2)

    ###beautification
    ax.legend(loc=0, title=r'$N=20 \quad n=10 \quad trials=100000$', fontsize=12)
    ax.set_ylabel(r'$MSE$')
    ax.set_xlabel("Kurtosis")
    ax.grid()

    ###putting the plot
    plt.show()
