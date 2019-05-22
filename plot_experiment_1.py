if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import os

    csv = False

    data = open('results_mse_def|50|100|2|1000_100av.txt',
                'r').readlines()
    x, y1, y2 = [int(el.split('|')[0]) for el in data[1:]], \
                [float(el.split('|')[1]) for el in data[1:]], float(data[0].split('|')[1])

    ###defining your fitfunction

    def func(x, a, b):
        return a + (b / x)

    ###let us guess some start values
    initialGuess = [0.2, 0.02]
    guessedFactors = [func(n, *initialGuess) for n in x]

    ###making the actual fit
    popt, pcov = curve_fit(func, x, y1)

    #one may want to
    print(popt)
    print(pcov)

    ###preparing data for showing the fit
    # basketCont = np.linspace(min(x), max(x))
    fittedData = [func(n, *popt) for n in x]

    ###preparing the figure
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1, 1, 1)

    ###the three sets of data to plot
    ax.plot(x, np.repeat(y2, len(y1)), color='b', label="Without Bagging")
    ax.plot(x, y1, linestyle='', marker='o', color='r', label="With Bagging")
    # ax.plot(x, guessedFactors, linestyle='',marker='^', color='b', label="initial guess")
    ax.plot(x, fittedData, linestyle='-', color='#900000', label="fit with ({0:0.2g},{1:0.2g})".format(*popt))

    ###beautification
    ax.legend(loc=0, title=r'$\sigma=5 \quad X \in R^{1000, 2}$', fontsize=12)
    ax.set_ylabel("MSE")
    ax.set_xlabel("N")
    ax.grid()

    ###putting the plot
    plt.show()

    fig1.savefig(os.path.join('plots', f'sigma=5_linreg_N=100_test_size=095_1000samples.png'))
