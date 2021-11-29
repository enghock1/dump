import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold

def HistogramOfProjections(dist, ytrue, title=None, numbin=20, savefig=False):
    ######
    # Histogram of Projection function
    # argument dist [numpy [n,]]: euclidean distance from decision boundary for n data points

    # argement ytrue [numpy [n,]]: binary label for n data points.
    #                              The index of data points should corresponding to
    #                              the same data points in arguement dist

    # numbin [int]: number of bin (default = 20)
    ######

    assert np.unique(ytrue).shape[0] == 2  # assert binary class label

    cls1, cls2 = np.unique(ytrue)
    cls1_index = np.where(ytrue==cls1)
    cls2_index = np.where(ytrue==cls2)

    cls1_dist  = dist[cls1_index]
    cls2_dist  = dist[cls2_index]

    # Perform histograming
    cls1_binedge = np.linspace(np.min(cls1_dist), np.max(cls1_dist), numbin)
    cls2_binedge = np.linspace(np.min(cls2_dist), np.max(cls2_dist), numbin)
    cls1_hist, cls1_hist_dist = np.histogram(cls1_dist,cls1_binedge)
    cls2_hist, cls2_hist_dist = np.histogram(cls2_dist,cls2_binedge)
    cls1_hist_dist = cls1_hist_dist + np.abs(np.abs(cls1_hist_dist[0]) - np.abs(cls1_hist_dist[1]))/2
    cls1_hd = cls1_hist_dist[:-1].copy()
    cls2_hist_dist = cls2_hist_dist + np.abs(np.abs(cls2_hist_dist[0]) - np.abs(cls2_hist_dist[1]))/2
    cls2_hd = cls2_hist_dist[:-1].copy()

    # Plot figure
    plt.figure(figsize=(8,6))
    max_y = np.amax(np.maximum(cls1_hist,cls2_hist))

    plt.scatter(dist, np.ones(dist.shape[0])*max_y/2, c=ytrue, s=30, cmap=plt.cm.Paired)
    plt.plot(cls1_hd, cls1_hist, cls2_hd, cls2_hist)
    plt.plot(np.zeros(100), np.linspace(-10,max_y+20,100),'k')
    plt.plot(np.zeros(100)-1, np.linspace(-10,max_y+20,100),'k--')
    plt.plot(np.zeros(100)+1, np.linspace(-10,max_y+20,100),'k--')

    plt.ylim((0, max_y+20))
    if title == None:
        plt.title('Histogram of Projections', fontsize='xx-large')

        if savefig:
            plt.savefig('hop.png')
    else:
        plt.title(title, fontsize='xx-large')

        if savefig:
            plt.savefig(title+'.png')


def plot_HOP(model, X, y, title=None, savefig=False):
    ## Plot histogram of projection
    # Input Argument:
    # model [Sklearn.SVM class] = trained SVM model
    # X [numpy array [n,d]] = input X
    # y [numpy array [n,]] = output y
    # title [string] = title for HOP

    dist_ypred = model.decision_function(X)
    HistogramOfProjections(dist_ypred, y, title=title, savefig=savefig)


def train_SVC(trnX, trny, valX, valy, tstX, tsty, Cs, Kps, kernel_type='rbf', weight=None, HOP=False):
    # train support vector classification

    # setup variables
    val_errors = np.zeros((len(Cs),len(Kps)))

    # perform model selection
    # for each parameter C
    for i, C in enumerate(Cs):
        # for each parameter Kp
        for j, kp in enumerate(Kps):

            # model fitting
            if kernel_type == 'rbf':
                model = SVC(C=C, kernel=kernel_type, gamma=kp, class_weight=weight)
            elif kernel_type == 'poly':
                model = SVC(C=C, kernel=kernel_type, degree=kp, class_weight=weight)

            model.fit(trnX, trny)

            # model validating
            ypred = model.predict(valX)
            val_errors[i,j] = zero_one_loss(valy, ypred)

    # find optimal parameters with smallest validation error
    optC_idx, optkp_idx = np.where(val_errors == val_errors.min())
    optC, optkp = Cs[optC_idx[0]], Kps[optkp_idx[0]]

    # train optimal model
    if kernel_type == 'rbf':
        opt_model = SVC(C=optC, kernel=kernel_type, gamma=optkp, class_weight=weight)
    elif kernel_type == 'poly':
        opt_model = SVC(C=optC, kernel=kernel_type, degree=optkp, class_weight=weight)
    opt_model.fit(trnX, trny)

    # get training Error
    trn_ypred = opt_model.predict(trnX)
    trn_error = zero_one_loss(trny, trn_ypred)

    # get test Error
    tst_ypred = opt_model.predict(tstX)
    tst_error = zero_one_loss(tsty, tst_ypred)

    # generate training and test HOP figure
    if weight==None:
        train_name = 'Standard Training HOP'
        test_name = 'Standard Test HOP'
    else:
        train_name = 'Imbalance Training HOP'
        test_name = 'Imbalance Test HOP'

    if HOP:
        plot_HOP(opt_model, trnX, trny, title = train_name, savefig=True)
        plot_HOP(opt_model, tstX, tsty, title = test_name, savefig=True)

    return opt_model, trn_error, tst_error, optC, optkp


def train_KNN(trnX, trny, valX, valy, tstX, tsty, Ks):
    # train k-nearest neighbor classification

    # setup variables
    val_errors = np.zeros((len(Ks),))

    # perform model selection
    # for each parameter k
    for i, k in enumerate(Ks):

            # model fitting
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(trnX, trny)

            # model validating
            ypred = model.predict(valX)
            val_errors[i] = zero_one_loss(valy, ypred)

    # find optimal parameters with smallest validation error
    optk_idx = np.where(val_errors == val_errors.min())
    optk = Ks[optk_idx[0][0]]

    # train optimal model
    opt_model = KNeighborsClassifier(n_neighbors=optk)
    opt_model.fit(trnX, trny)

    # get training Error
    trn_ypred = opt_model.predict(trnX)
    trn_error = zero_one_loss(trny, trn_ypred)

    # get test Error
    tst_ypred = opt_model.predict(tstX)
    tst_error = zero_one_loss(tsty, tst_ypred)

    return opt_model, trn_error, tst_error, optk


def plot_margin(clf, X, y, grid_size=(200,200), background_color=True,
                title='Decision Boundary', fig_name='boundary.png', algo='svm'):

    plt.figure(figsize=(8,6))

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    plt.xlabel(r'$x_1$', fontsize='xx-large')
    plt.ylabel(r'$x_2$', fontsize='xx-large')

    plt.title(title, fontsize='xx-large')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_size[0]),
                         np.linspace(ylim[0], ylim[1], grid_size[1]))

    # convert to two-dimensional
    X_plot = np.vstack([xx.ravel(), yy.ravel()]).T

    # obtain prediction from the model
    if algo=='svm':
        Y = clf.decision_function(X_plot).reshape(xx.shape)

        sv = clf.support_
        ax.scatter(X[sv,0], X[sv,1], s=100,
                   linewidth=1, facecolors='none', edgecolors='k')

        contours = plt.contour(xx, yy, Y, colors='k',
                               levels=[-1,0,1], alpha=0.5,
                               linestyles=['--', '-', '--'])
    else:
        Y = clf.predict(X_plot).reshape(xx.shape)

        contours = plt.contour(xx, yy, Y, colors='k',
                               levels=[0], alpha=0.5,
                               linestyles=['-'])

    if background_color == True:
        plt.imshow(Y, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
               origin='lower', cmap=plt.cm.PuOr)




    plt.savefig(fig_name)


### Hyperbola dataset functions
def cls0(Min, Max, n, s=0.225):
    x = np.random.uniform(Min, Max, (n,1))
    noise = np.random.normal(0, 0.03, (n,1))
    #y = ((x-0.4)*3)**2 + s + noise
    y = ((x-0.4)*3)**2 + s + noise

    return np.concatenate((x,y), axis=1)


def cls1(Min, Max, n, s=0.225):
    x = np.random.uniform(Min, Max, (n,1))
    noise = np.random.normal(0, 0.03, (n,1))
    #y = 1 - ((x-0.6)*3)**2 - s + noise
    y = 1 - ((x-0.6)*3)**2 - s + noise

    return np.concatenate((x,y), axis=1)


def p3_high_d(n):

    X = np.random.uniform(size=(n, 20))
    y = np.zeros((n,))

    idx = [i for i in range(1,20,2)]
    for i in range(n):
        Sum = 0
        for j in idx:
            Sum += X[i,j]
        if Sum > 5:
            y[i] = 1
        else:
            y[i] = -1

    return X, y


def cls_digit(digit, y_value, X_all, y_all):
    # get specific digit data from MNIST dataset

    idx = y_all==digit
    X_digit = X_all[idx,:]
    y_digit = np.ones((X_digit.shape[0],))*y_value

    return X_digit, y_digit


def cls_trn_val_tst(X, y, ntrn, nval, ntst):
    # get training, validation, and test set

    trn_X = np.zeros((ntrn, X.shape[1]))
    val_X = np.zeros((nval, X.shape[1]))
    tst_X = np.zeros((ntst, X.shape[1]))

    trn_y = np.zeros((ntrn,))
    val_y = np.zeros((nval,))
    tst_y = np.zeros((ntst,))

    n = ntrn + nval + ntst
    idx = np.random.choice(X.shape[0], n, replace=False)

    i = 0
    if ntrn != 0:
        trn_X = X[idx[i:i+ntrn],:]
        trn_y = y[idx[i:i+ntrn]]
        i += ntrn

    if nval != 0:
        val_X = X[idx[i:i+nval],:]
        val_y = y[idx[i:i+nval]]
        i+= nval

    if ntst != 0:
        tst_X = X[idx[i:i+ntst],:]
        tst_y = y[idx[i:i+ntst]]

    return trn_X, trn_y, val_X, val_y, tst_X, tst_y


def combine_cls(X1, y1, X2, y2, output_noise_percentage=0):

    y1_cp = y1.copy()
    y2_cp = y2.copy()

    if output_noise_percentage > 0:
        cls1 = np.unique(y1_cp)
        cls2 = np.unique(y2_cp)

        idx_y1 = np.random.choice(y1_cp.shape[0], int(y1_cp.shape[0]*output_noise_percentage), replace=False)
        idx_y2 = np.random.choice(y2_cp.shape[0], int(y2_cp.shape[0]*output_noise_percentage), replace=False)

        y1_cp[idx_y1] = cls2
        y2_cp[idx_y2] = cls1

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1_cp, y2_cp))

    return X, y


def get_MNIST_data(ntrn, nval, ntst, noise=0):

    # get MNIST data
    data = pd.read_csv('C:\\Users\\leex7132\\Documents\\Datasets\\MNIST\\mnist_train.csv')
    X = np.array(data.iloc[:,1:])
    y = np.array(data.iloc[:,0])

    # class label
    cls_neg = 5
    cls_pos = 8

    X_n, y_n = cls_digit(cls_neg, -1, X, y)
    X_p, y_p = cls_digit(cls_pos, 1, X, y)

    trn_X_n, trn_y_n, val_X_n, val_y_n, tst_X_n, tst_y_n = cls_trn_val_tst(X_n, y_n, ntrn, nval, ntst)
    trn_X_p, trn_y_p, val_X_p, val_y_p, tst_X_p, tst_y_p = cls_trn_val_tst(X_p, y_p, ntrn, nval, ntst)

    trn_X, trn_y = combine_cls(trn_X_n, trn_y_n, trn_X_p, trn_y_p, output_noise_percentage=noise)
    val_X, val_y = combine_cls(val_X_n, val_y_n, val_X_p, val_y_p, output_noise_percentage=noise)
    tst_X, tst_y = combine_cls(tst_X_n, tst_y_n, tst_X_p, tst_y_p, output_noise_percentage=noise)

    trn_X, val_X, tst_X = trn_X/255, val_X/255, tst_X/255

    return trn_X, val_X, tst_X, trn_y, val_y, tst_y


def problem2():

    # sample size
    ntrn = 100
    nval = 100
    ntst = 2000

    # train data
    class0 = cls0(0.2, 0.6, ntrn//2)
    class1 = cls1(0.4, 0.8, ntrn//2)
    trnX = np.concatenate((class0, class1), axis=0)
    trny = np.concatenate((np.ones((ntrn//2,))*-1, np.ones((ntrn//2,))))

    # validation data
    class0 = cls0(0.2, 0.6, nval//2)
    class1 = cls1(0.4, 0.8, nval//2)
    valX = np.concatenate((class0, class1), axis=0)
    valy = np.concatenate((np.ones((nval//2,))*-1, np.ones((nval//2,))))

    # test data
    class0 = cls0(0.2, 0.6, ntst//2)
    class1 = cls1(0.4, 0.8, ntst//2)
    tstX = np.concatenate((class0, class1), axis=0)
    tsty = np.concatenate((np.ones((ntst//2,))*-1, np.ones((ntst//2,))))

    Cs = [2**i for i in range(-2,5)]
    Kps = [i for i in range(48,104,8)]

    ## standard SVC
    opt_model, trn_error, tst_error, optC, optkp = train_SVC(trnX, trny,
                                                             valX, valy,
                                                             tstX, tsty,
                                                             Cs, Kps,
                                                             kernel_type='rbf',
                                                             HOP=True)
    print('Standard SVM')
    print('Training Error: ', trn_error, ' Test Error: ', tst_error, ' Opt C: ', optC, ' Opt Gamma: ', optkp)

    # plot margin
    plot_margin(opt_model, trnX, trny, title='Standard SVM', fig_name='standard.png')

    ## Imbalance SVC
    weight = {1:2}
    opt_model, trn_error, tst_error, optC, optkp = train_SVC(trnX, trny,
                                                             valX, valy,
                                                             tstX, tsty,
                                                             Cs, Kps,
                                                             kernel_type='rbf',
                                                             weight=weight,
                                                             HOP=True)
    print('Imbalance SVM')
    print('Training Error: ', trn_error, ' Test Error: ', tst_error, ' Opt C: ', optC, ' Opt Gamma: ', optkp)

    # plot margin
    plot_margin(opt_model, trnX, trny, title='Imbalance SVM', fig_name='imbalance.png')


def problem3():

    # Ripley data
    print('Ripley data')
    trn_data = np.array(pd.read_csv('ripley_trn.csv'))
    tst_data = np.array(pd.read_csv('ripley_tst.csv'))

    # separate a section of tst_data as val_data
    nval = 250
    cls1_val = np.random.choice(range(tst_data.shape[0]//2),nval//2, replace=False)
    cls2_val = np.random.choice(range(tst_data.shape[0]//2,tst_data.shape[0]), nval//2, replace=False)
    val_cls1 = tst_data[cls1_val,:]
    val_cls2 = tst_data[cls2_val,:]
    val_data = np.concatenate((val_cls1, val_cls2), axis=0)

    tst_data = np.delete(tst_data, np.concatenate((cls1_val, cls2_val)), 0)

    trnX = trn_data[:,:2]
    trny = trn_data[:,2]
    valX = val_data[:,:2]
    valy = val_data[:,2]
    tstX = tst_data[:,:2]
    tsty = tst_data[:,2]

    ## perform KNN
    Ks = [3,5,7,9,11,13,15]
    opt_model, trn_error, tst_error, optk = train_KNN(trnX, trny, valX, valy, tstX, tsty, Ks)

    plot_margin(opt_model, trnX, trny, title='k-NN', fig_name='knn.png', algo='knn')

    print('KNN')
    print('Training Error: ', trn_error, ' Test Error: ', tst_error, ' Opt k: ', optk)


    ## perform poly-SVM
    Cs = [2**i for i in range(-2,5)]
    Ds = [1,2,3,4]
    opt_model, trn_error, tst_error, optC, optkp = train_SVC(trnX, trny,
                                                             valX, valy,
                                                             tstX, tsty,
                                                             Cs, Ds,
                                                             kernel_type='poly')

    plot_margin(opt_model, trnX, trny, title='Poly SVM', fig_name='svm.png')

    print('Poly SVM')
    print('Training Error: ', trn_error, ' Test Error: ', tst_error, ' Opt C: ', optC, ' Opt degree: ', optkp)


    # High dimensional dataset
    print('High Dimensional Data')
    ntrn = 50
    nval = 50
    ntst = 1000
    trnX, trny = p3_high_d(ntrn)
    valX, valy = p3_high_d(nval)
    tstX, tsty = p3_high_d(ntst)

    ## perform KNN
    Ks = [3,4,5,7,9,11,13,15]
    opt_model, trn_error, tst_error, optk = train_KNN(trnX, trny, valX, valy, tstX, tsty, Ks)
    print('KNN')
    print('Training Error: ', trn_error, ' Test Error: ', tst_error, ' Opt k: ', optk)

    ## perform poly-SVM
    Cs = [2**i for i in range(-2,5)]
    Ds = [1,2,3,4]
    opt_model, trn_error, tst_error, optC, optkp = train_SVC(trnX, trny,
                                                             valX, valy,
                                                             tstX, tsty,
                                                             Cs, Ds,
                                                             kernel_type='poly')
    print('Poly SVM')
    print('Training Error: ', trn_error, ' Test Error: ', tst_error, ' Opt C: ', optC, ' Opt degree: ', optkp)


def problem4():

    ## a) small dataset problem
    n_realization = 100
    ntrn = 10
    nval = 10
    ntst = 1000

    avg_trn_error = np.zeros((n_realization,))
    avg_tst_error = np.zeros((n_realization,))
    for n in range(n_realization):

        trnX, valX, tstX, trny, valy, tsty = get_MNIST_data(ntrn, nval, ntst)

        # parameter range
        Cs = [2**i for i in range(-2,5)]
        Kps = [2**i for i in range(-2,5)]

        # rbf SVC
        opt_model, trn_error, tst_error, optC, optkp = train_SVC(trnX, trny,
                                                                 valX, valy,
                                                                 tstX, tsty,
                                                                 Cs, Kps,
                                                                 kernel_type='rbf',
                                                                 HOP=False)
        avg_trn_error[n] = trn_error
        avg_tst_error[n] = tst_error
    print('Small dataste SVM')
    print('Average Training Error: ', np.mean(avg_trn_error), ' Average Test Error: ', np.mean(avg_tst_error))
    print('Std Training Error: ', np.std(avg_trn_error), ' Std Test Error: ', np.std(avg_tst_error))

    ## b) large dataset problem
    ntrn = 400
    nval = 400
    ntst = 1000

    trnX, valX, tstX, trny, valy, tsty = get_MNIST_data(ntrn, nval, ntst)

    # parameter range
    Cs = [2**i for i in range(-2,5)]
    Kps = [2**i for i in range(-2,5)]

    # rbf SVC
    opt_model, trn_error, tst_error, optC, optkp = train_SVC(trnX, trny,
                                                             valX, valy,
                                                             tstX, tsty,
                                                             Cs, Kps,
                                                             kernel_type='rbf',
                                                             HOP=False)
    print('Large dataste SVM')
    print('Training Error: ', trn_error, ' Test Error: ', tst_error, ' Opt C: ', optC, ' Opt Gamma: ', optkp)

    np.savetxt('b_trnX.csv', trnX, delimiter=',')
    np.savetxt('b_valX.csv', valX, delimiter=',')
    np.savetxt('b_tstX.csv', tstX, delimiter=',')
    np.savetxt('b_trny.csv', trny, delimiter=',')
    np.savetxt('b_valy.csv', valy, delimiter=',')
    np.savetxt('b_tsty.csv', tsty, delimiter=',')

    ## c) large dataset problem with 5% noise
    ntrn = 400
    nval = 400
    ntst = 1000

    trnX, valX, tstX, trny, valy, tsty = get_MNIST_data(ntrn, nval, ntst, noise=0.05)

    # parameter range
    Cs = [optC]
    Kps = [optkp]

    # rbf SVC
    opt_model, trn_error, tst_error, optC, optkp = train_SVC(trnX, trny,
                                                             valX, valy,
                                                             tstX, tsty,
                                                             Cs, Kps,
                                                             kernel_type='rbf',
                                                             HOP=True)
    print('Large dataste SVM with 5% output noise')
    print('Training Error: ', trn_error, ' Test Error: ', tst_error, ' Opt C: ', optC, ' Opt Gamma: ', optkp)

    np.savetxt('c_trnX.csv', trnX, delimiter=',')
    np.savetxt('c_valX.csv', valX, delimiter=',')
    np.savetxt('c_tstX.csv', tstX, delimiter=',')
    np.savetxt('c_trny.csv', trny, delimiter=',')
    np.savetxt('c_valy.csv', valy, delimiter=',')
    np.savetxt('c_tsty.csv', tsty, delimiter=',')

def main():

    # run problem 2
    #problem2()

    # run problem 3
    #problem3()

    # run problem 4 - SVM
    problem4()


if __name__ == "__main__":
    main()
