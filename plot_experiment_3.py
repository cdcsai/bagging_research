import matplotlib.pyplot as plt
import os


def plot_kurt(path, save_path='plots'):
    data = open(path, 'r').readlines()
    k = [float(el.split('|')[0]) for el in data]
    mse_var = [float(el.split('|')[1]) for el in data]
    mse_var_bag = [float(el.split('|')[2]) for el in data]

    # N, n, trials, a = 20, 10, 100000, 0.125

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
    fig1.savefig(os.path.join(save_path, f'N={N}_n={n}_trials={trials}_a={str(a)[2:]}.png'))


if __name__ == '__main__':
    plot_kurt('res_special_rad__a=0.125_trials=10000_N=50_n=10.txt', save_path='plots')
