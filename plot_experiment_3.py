import matplotlib.pyplot as plt


def plot_kurt(path):
    data = open(path, 'r').readlines()
    k = [float(el.split('|')[0]) for el in data]
    mse_var = [float(el.split('|')[1]) for el in data]
    mse_var_bag = [float(el.split('|')[2]) for el in data]

    N, n, trials, a = int(path.split('=')[-2].split('_')[0]), \
                   int(path.split('=')[-1].split('.')[0]), \
                   int(path.split('=')[-3].split('_')[0]),\
                      round(float(path.split('=')[-4].split('_')[0]),3)

    # preparing the figure
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1, 1, 1)

    # the three sets of data to plot
    ax.plot(k, mse_var, linestyle='', marker='o', color='r', label=r'MSE without Bagging')
    ax.plot(k, mse_var_bag, linestyle='', marker='o', color='b', label=r'MSE with Bagging')
    plt.axvline(x=3 / 2)

    # beautification
    ax.legend(loc=0, title=f'N={N} n={n} trials={trials} a={a}', fontsize=12)
    ax.set_ylabel(r'$MSE$')
    ax.set_xlabel("Kurtosis")
    ax.grid()

    # putting the plot
    plt.show()


if __name__ == '__main__':
    plot_kurt('res_special_rad__a=0.0625_trials=1000_N=100_n=10.txt')
