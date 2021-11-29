import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt

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
