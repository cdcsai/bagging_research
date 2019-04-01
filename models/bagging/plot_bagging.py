def special_bool(boolean: str):
    if type(boolean) == bool:
        return boolean
    else:
        if boolean == "True":
            return True
        else:
            return False

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from collections import defaultdict
    bagging, N, T, acc = [], [], [], []
    with open('/home/charles/Desktop/deep_nlp_research/models/bagging/LogReg/results_bagging_logreg.txt') as f:
        for i, line in enumerate(f.readlines()):
            split_line = line.split('|')
            bagging.append(int(special_bool(split_line[0])))
            N.append(int(split_line[1]))
            T.append(float(split_line[2]))
            acc.append(float(split_line[3]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sp = ax.scatter(N, T, bagging, s=100, c=acc)
    ax.set_xlabel('N')
    ax.set_ylabel('T')
    ax.set_zlabel('Bagging or Not')
    plt.colorbar(sp)
    plt.show()
