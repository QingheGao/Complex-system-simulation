import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def beta():
    """plot of average minimum sigma* value (y) vs beta (x) for each model"""
    # for one Model
    sns.set()
    beta = ['0','05','1','5','inf']
    for model in ['m1']:
        sigmastar = []
        for b in ['0','05','1','5','inf']:
            runs = pickle.load(open(f"results/{model}_tau_{b}.pkl", 'rb'))
            total = 0
            for run in runs.items():
                total += run[1][-1] / len(run[1])
            sigmastar.append(total/5)
        df = pd.DataFrame({'beta':beta, 'sigmastar':sigmastar})
        sns.lineplot(x="beta", y="sigmastar", data=df,label=model)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\tau$")
    plt.legend()
    plt.savefig("plots/output.png")
    return

def loglog():
    """plot of log(sigma*) value (y) vs log(time) (x) for each model"""
    sns.set()
    f, ax = plt.subplots()
    ax.set(xscale='log', yscale='log')
    plt.ylim([0.01,10])

    for i in ['1','2','3','4']:
        sigmastar = pickle.load(open(f"results/m{i}_sigmastar_inf.pkl", 'rb'))[1]
        t = np.arange(0,len(sigmastar))
        df = pd.DataFrame({'time':t, 'sigmastar':sigmastar})
        sns.lineplot('time', 'sigmastar', data=df, ax=ax, label=i)
    plt.xlabel('Time')
    plt.ylabel(r"$\sigma^*$")
    plt.legend()
    plt.savefig("plots/loglog_all_inf.png")
    return

def heatmap():
    """heatmap of distribution of ants/brood care at end of run"""
    pass

def make_dataframe_sigmastar():
    model = ['1'] * 25# + (['2'] * 25) + (['3'] * 25) + (['4'] * 25) + (['5'] * 25)
    beta = ['0','0','0','0','0','0.5','0.5','0.5','0.5','0.5',
            '1','1','1','1','1','5','5','5','5','5','10','10','10','10','10']# * 5
    min_sigmastar = []
    data = read_data_sigmastar()
    df = pd.DataFrame({'model':model,'beta':beta,'sigmastar':data})
    return df

def read_data_sigmastar():
    data = []
    models = ['m1']
    betavalues = ['0','05','1','5','inf']
    for model in models:
        for beta in betavalues:
            sigmas = pickle.load(open(f"results/{model}_sigmastar_{beta}.pkl", 'rb'))
            for i in sigmas.items():
                data.append(min(i[1]))
    return data


# matrix plot
def plot_matrix(data,norm,beta,model):
	""" Plot the tau matrix as a heatmap. """

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
		im.set_cmap("summer")
		im.set_clim(0,0.7)
		ax.flat[idx].axis('off')

		print('Maximum',max(map(max, matrix))) # Check whether the maximum is ever larger than clim 0.7

	# Make the colorbar
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.4, 0.05, 0.20])
	fig.colorbar(im, cax=cbar_ax)

	plt.savefig(f"plots/m{model}_matrix_{beta}.png",dpi=400,bbox_inches='tight')
	plt.show()

def read_matrix():
	""" Read the tau matrix for all models and betas. """

	# Loop over four models and over beta to save and plot data
	for model in range(1,5):
		for beta in ['0','05','1','5','inf']:
			data = pickle.load(open(f"results/m{model}_matrix_{beta}.pkl", 'rb'))
			norm = pickle.load(open(f"results/m{model}_tau_{beta}.pkl", 'rb'))

			# Plot the matrix each of the five runs
			plot_matrix(data,norm,beta,model)

beta()
