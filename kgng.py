#---------------------------------------
#Since : Jun/17/2012
#UpdatBe: 2013/05/15
# -*- coding: utf-8 -*-
# Using Growing Neural Gas 1995
#---------------------------------------
from PIL import Image
import numpy as np
import math as mt
import pylab as pl
import networkx as nx
import sys
from scipy import ndimage
from sklearn import cluster, datasets
import matplotlib.pyplot as plt

from gng import GNG

class kernel_GNG(GNG):
    def __init__(self, num = 25, end = 100000, lam = 100, ew = 0.1, en = 0.01, amax = 20.0, alpha = 0.5, beta = 0.9,  kernel_func = "gng", param = 0.1):
        super().__init__(num, end, lam, ew, en, amax, alpha, beta)

        self.epsilon = 1e-7

        self.kernel_func = kernel_func
        self.param = param

    def dists(self, x, units):
        #calculate distance

        if self.kernel_func == "rbf":
            return 2 * (1 - np.exp(-np.linalg.norm(x - units, axis = 1)**2 / (2 * (self.param**2))) )
        elif self.kernel_func == "lap":
            return 2 * (1 - np.exp(-np.linalg.norm(x - units, axis = 1) / self.param) )
        elif self.kernel_func == "pol":
            return (np.dot(x, x) + 1)**self.param - 2 *((np.dot(units, x) + 1)**self.param) + (np.diag(np.dot(units, units.T)) + 1)**self.param
        elif self.kernel_func == "mq":
            return 2 * (self.param - np.sqrt(np.linalg.norm(x - units, axis = 1)**2 + self.param**2))
        elif self.kernel_func == "imq":
            return 2 * (1/self.param - 1/np.sqrt(np.linalg.norm(x - units, axis = 1)**2 + self.param**2))
        elif self.kernel_func == "pow":
            return 2 * np.linalg.norm(x - units, axis = 1)**self.param
        elif self.kernel_func == "log":
            return 2 * np.log(np.linalg.norm(x - units, axis = 1)**self.param + 1)
        elif self.kernel_func == "cau":
            return 2 * (1 - 1 / (1 + np.linalg.norm(x - units, axis = 1)**2 / (self.param**2)))
        elif self.kernel_func == "gng":
            return np.linalg.norm(units - x, axis=1)**2
        else:
            print("Error")
            exit()

    def dw(self, x, unit):
        if self.kernel_func == "rbf":
            return (x - unit) / (self.param**2) * np.exp(- np.linalg.norm(x - unit)**2/ (2 * (self.param**2)))
        elif self.kernel_func == "lap":
            return (x - unit) * np.exp(-np.linalg.norm(x - unit)/self.param) / self.param / (np.linalg.norm(x - unit)  + self.epsilon) # avoid zero-devided
        elif self.kernel_func == "pol":
            return - self.param * (unit * ((np.dot(unit, unit) + 1)**(self.param - 1)) - x * ((np.dot(unit, x) + 1)**(self.param - 1)))
        elif self.kernel_func == "mq":
            return 2 * (x - unit) / np.sqrt(np.linalg.norm(x - unit)**2 + self.param**2)
        elif self.kernel_func == "imq":
            return (x - unit) / ((np.linalg.norm(x - unit)**2 + self.param**2)**1.5)
        elif self.kernel_func == "pow":
            return (x - unit) * self.param * (np.linalg.norm(x - unit)**(self.param - 2))
        elif self.kernel_func == "log":
            return self.param * (x - unit) * np.linalg.norm(x - unit)**(self.param - 2) / (np.linalg.norm(x - unit)**self.param + 1)
        elif self.kernel_func == "cau":
            return (x - unit) * 2 / (self.param**2) / ((1 + np.linalg.norm(x - unit)**2 / (self.param**2))**2)
        elif self.kernel_func == "gng":
            return x - unit
        else:
            print("Error")
            exit()


if __name__ == '__main__':

    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    gng = kernel_GNG()

    gng.train(noisy_circles[0])
    plt.scatter(noisy_circles[0][:,0], noisy_circles[0][:,1])


    nx.draw_networkx_nodes(gng.g_units,gng.units,node_size=50,node_color=(0.5,1,1))
    nx.draw_networkx_edges(gng.g_units,gng.units,width=2,edge_color='b',alpha=0.5)

    plt.savefig("gng.png")

# if __name__ == '__main__':
#     np.random.seed(1)

#     title = "Blobs"#["Square", "Blobs", "Circles", "Moons", "Iris", "Wine", "digits"]

#     som = kernel_GNG(kernel_func = "non", lam = 0.5)

#     k, data, true_labels = dataset.dataset(title)

#     som.train(data)

#     nodes = nx.number_of_nodes(som.g_units)
#     edges = 2 * nx.number_of_edges(som.g_units) / float(nx.number_of_nodes(som.g_units))
#     clustering = nx.average_clustering(som.g_units)
#     t = som.t

#     print(title, t, nodes, edges, clustering)

#     plt.scatter(data[:,0], data[:,1], s = 5, c="gray")

#     nx.draw_networkx_nodes(som.g_units,som.units,node_size=10,node_color="k")
#     nx.draw_networkx_edges(som.g_units,som.units,width=1, edge_color='k')

#     #plt.show()
#     plt.savefig("sig0.5_blobs.eps")
