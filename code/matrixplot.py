import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_matrix(data,norm,beta,model):
	""" Plot the tau matrix as a heatmap for five runs in one figure. """

	fig,ax = plt.subplots(1, 5)
	
	# Plot the matrix for each run
	for idx,runs in enumerate(data.items()):

		normalize = len(norm[idx+1])

		# Normalize for each run by dividing over total timesteps
		matrix = np.zeros((25,25))
		for i in range(25):
			for j in range(25):
				matrix[i][j] = runs[1][i][j] / normalize

		# Make the matrixplot
		im = ax.flat[idx].imshow(matrix)
		im.set_cmap("PiYG")
		im.set_clim(0,0.7)
		ax.flat[idx].axis('off')

		print('Maximum',max(map(max, matrix))) # Check whether the maximum is ever larger than clim 0.7

	# Make the colorbar
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.4, 0.05, 0.20])
	fig.colorbar(im, cax=cbar_ax)
	plt.suptitle(f'model-{model} beta-{beta}')
	plt.savefig(f"plots/m{model}_matrix_{beta}.png",dpi=400,bbox_inches='tight')
	plt.show()

def plot_indiv_matrix(data,norm,beta,model):
	""" Plot the tau matrix as a heatmap, one run per figure. """
	if beta == "05":
		b = "0.5"
	elif beta == "inf":
		b = r"$\infty$"
	else:
		b = beta
	
	# Plot the matrix for the first run
	normalize = len(norm[1])

	# Normalize for each run by dividing over total timesteps
	matrix = np.zeros((25,25))
	for i in range(25):
		for j in range(25):
			matrix[i][j] = data[1][i][j] / normalize

	# Make the matrixplot
	im = plt.imshow(matrix)
	im.set_cmap("summer")
	im.set_clim(0,0.7)
	cbar = plt.colorbar()
	cbar.ax.set_ylabel(r'$\tau_{norm}$')
	plt.axis('off')

	print('Maximum',max(map(max, matrix))) # Check whether the maximum is ever larger than clim 0.7

	plt.title(rf'Model {model}, $\beta$ ={b}')
	plt.savefig(f"plots/indiv_m{model}_matrix_{beta}.png",dpi=400,bbox_inches='tight')
	plt.show()

def read_matrix():
	""" Read the tau matrix for all models and betas. """

	# Loop over four models and over beta to save and plot data
	for model in range(1,5):
		for beta in ['0','05','1','5','inf']:
			data = pickle.load(open(f"results/model{model}/m{model}_matrix_{beta}.pkl", 'rb'))
			norm = pickle.load(open(f"results/model{model}/m{model}_tau_{beta}.pkl", 'rb'))

			# Plot the matrix each of the five runs
			# plot_matrix(data,norm,beta,model) # Make a plot for five runs in one figure
			# plot_indiv_matrix(data,norm,beta,model) # Plot one run for each model

read_matrix()

