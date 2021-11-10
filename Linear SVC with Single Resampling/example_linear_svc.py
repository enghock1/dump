import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import train_test_split, KFold

__date__ = '11/10/2021'
__author__ = 'Eng Hock Lee'
__email__ = 'leex7132@umn.edu'

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


def plot_decision_boundary(X, y, model):
    # plot decision boundary

    # get the separating hyperplane
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(X[:,0].min(), X[:,0].max())
    yy = a * xx - (model.intercept_[0]) / w[1]
    margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    plt.figure(figsize=(8, 6))
    plt.clf()
    plt.plot(xx, yy, "k-")
    plt.plot(xx, yy_down, "k--")
    plt.plot(xx, yy_up, "k--")
    plt.title('Decision Boundary', fontsize=16)

    plt.scatter(X[:,0], X[:,1], c=y)
    plt.savefig('boundary.png')


def train_linear_SVC(X, y, Cs, k_fold):
    # linear SVC with single resampling technique

    # Input Argument:
    # X [numpy array [n,d]] = input X
    # y [numpy array [n,]] = output y
    # Cs [list] = list of parameter C
    # k_fold [int] = number of k fold cv

    # Output:
    # opt_model [sklearn.svc class] = optimal model
    # trn_error [float] = training error
    # tst_error [float] = test error
    # optC [float] = optimal C parameter

    # setup variables
    val_errors = np.zeros((len(Cs),))

    # Split data into training and test set
    trnX, tstX, trny, tsty = train_test_split(X, y, test_size=0.2)

    # perform single resampling technique for model selection
    kf = KFold(n_splits=k_fold)
    for lrn_idx, val_idx in kf.split(trnX):

        # split into learning and validation set
        lrnX, valX = trnX[lrn_idx,:], trnX[val_idx, :]
        lrny, valy = trny[lrn_idx], trny[val_idx]

        # for each parameter C
        for i, C in enumerate(Cs):

            # model fitting
            model = SVC(C=C, kernel='linear')
            model.fit(lrnX, lrny)

            # model validating
            ypred = model.predict(valX)
            val_errors[i] += zero_one_loss(valy, ypred)

    # divide val_errors by the number of k fold
    val_errors /= k_fold

    # find optimal parameters with smallest validation error
    optC_idx = np.where(val_errors == val_errors.min())
    optC = Cs[optC_idx[0][0]]

    # train optimal model
    opt_model = SVC(C=optC, kernel='linear')
    opt_model.fit(trnX, trny)

    # get training Error
    trn_ypred = opt_model.predict(trnX)
    trn_error = zero_one_loss(trny, trn_ypred)

    # get test Error
    tst_ypred = opt_model.predict(tstX)
    tst_error = zero_one_loss(tsty, tst_ypred)

    # generate training and test HOP figure
    plot_HOP(opt_model, trnX, trny, title = 'Training HOP', savefig=True)
    plot_HOP(opt_model, tstX, tsty, title = 'Test HOP', savefig=True)

    return opt_model, trn_error, tst_error, optC


def main():

    # generate randopm data for binary classification
    X, y = make_moons(n_samples=1000, noise=0.1)

    # parameters for linear SVC
    Cs = [10**i for i in range(-3,4)]

    # number of k fold for resampling
    k_fold = 5

    # perform linear SVC
    opt_model, trn_error, tst_error, optC = train_linear_SVC(X, y, Cs, k_fold)
    print('Optimal C: ', optC, ' Training Error: ', trn_error, ' Test Error: ', tst_error)

    # plot decision boundary
    plot_decision_boundary(X, y, opt_model)

if __name__ == "__main__":
    main()
