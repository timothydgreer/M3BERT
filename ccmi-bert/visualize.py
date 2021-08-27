import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import copy

import sys
import argparse


def plot_examples(cmap,data):
	fig,axs = plt.subplots(1,3,figsize = (6,3), constrained_layout=True)
	for [ax, i] in zip(axs, data):
		psm = ax.pcolormesh(i, cmap=cmap)
		fig.colorbar(psm, ax=ax)
	plt.show()

for i in range(int(sys.argv[2])):     
	x = np.load('./'+sys.argv[1]+'/example'+str(i)+'_original.npy')
	y = np.load('./'+sys.argv[1]+'/example'+str(i)+'_reconstructed.npy')
	z = np.load('./'+sys.argv[1]+'/example'+str(i)+'_input.npy')

	viridis = cm.get_cmap('viridis',256)
	   
	plot_examples(viridis, [np.transpose(x), np.transpose(y), np.transpose(z)])

