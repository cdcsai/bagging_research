if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv('/home/charles/Desktop/bagging_research/models/bagging/LinReg/non_lin_reg_2.csv')
    x, y = data['x'], data['y']

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit

    ###defining your fitfunction

    def func(x, a, b):
        return a + (b / x)

    ###let us guess some start values
    initialGuess = [0.2, 0.02]
    guessedFactors = [func(n, *initialGuess) for n in x]

    ###making the actual fit
    popt, pcov = curve_fit(func, x, y)

    #one may want to
    print(popt)
    print(pcov)

    ###preparing data for showing the fit
    # basketCont = np.linspace(min(x), max(x))
    fittedData = [func(n, *popt) for n in x]

    ###preparing the figure
    fig1 = plt.figure(1)
    ax=fig1.add_subplot(1,1,1)

    ###the three sets of data to plot
    ax.plot(x, y, linestyle='', marker='o', color='r', label="data")
    # ax.plot(x, guessedFactors, linestyle='',marker='^', color='b', label="initial guess")
    ax.plot(x, fittedData, linestyle='-', color='#900000', label="fit with ({0:0.2g},{1:0.2g})".format(*popt))

    ###beautification
    ax.legend(loc=0, title="graphs", fontsize=12)
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.grid()

    ###putting the plot
    plt.show()
