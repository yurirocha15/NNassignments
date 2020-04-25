import matplotlib.pyplot as plt
from YourAnswer import predictKNN
import numpy

def plotData(data):
    fig, ax = plt.subplots(figsize=(8,5))
    results_accepted = data[data.accepted == 1]
    results_rejected = data[data.accepted == 0]
    ax.scatter(results_accepted.test1, results_accepted.test2, marker='+', c='b', s=40)
    ax.scatter(results_rejected.test1, results_rejected.test2, marker='o', c='r', s=30)
    return ax

def vis_decision_boundary(x_tra, y_tra, k, typ='k--'):
    ax = plt.gca()

    lim0 = plt.gca().get_xlim()
    lim1 = plt.gca().get_ylim()
    
    x_ = numpy.linspace(lim0[0], lim0[1], 100)
    y_ = numpy.linspace(lim1[0], lim1[1], 100)
    xx, yy = numpy.meshgrid(x_, y_)
    
    pred = predictKNN(numpy.concatenate([xx.ravel()[:,None], yy.ravel()[:,None]], axis=1), x_tra, y_tra, k)
    
    ax.contourf(xx, yy, pred.reshape(xx.shape), cmap=plt.cm.coolwarm, alpha=0.4)

    ax.set_xlim(lim0)
    ax.set_ylim(lim1)