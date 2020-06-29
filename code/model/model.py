"""
In this file model initiation takes place (init function). Then, in the 'step' function,
is everything that should occur every timestep. Events at every timestep are now:
- ants move one step in a random direction of the Moore neighborhood
"""
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import pickle

from .agent import Ant, Brood,Fence

import numpy as np

WIDTH = 25
HEIGHT = 25
data_tau = []
data_sigma = []
data_sigmastar = []


class Anthill(Model):
    def __init__(self):

        self.grid = SingleGrid(WIDTH, HEIGHT, False)
        self.schedule = RandomActivation(self)
        self.running = True
        self.internalrate = 0.2
        self.ant_id = 1
        self.tau = np.zeros((WIDTH,HEIGHT))
        self.datacollector = DataCollector({"Total number of Ants": lambda m: self.get_total_ants_number(),
                                            "mean tau": lambda m: self.evaluation1(),
                                            "sigma": lambda m: self.evaluation2(),
                                            "sigma*" :  lambda m: self.evaluation3(),
                                            })

        # List containing all coordinates of the boundary, initial ants location and brood location
        self.bound_vals = []
        self.neigh_bound = []
        self.datacollector.collect(self)

        for i in range(WIDTH):
            for j in range(HEIGHT):
                if i == 0 or j == 0 or i == WIDTH-1 or j == HEIGHT-1:
                    self.bound_vals.append((i,j))
                if i == 1 or i == WIDTH - 2 or j == 1 or j == HEIGHT-2:
                    self.neigh_bound.append((i,j))

        # Make a Fence boundary
        b = 0
        for h in self.bound_vals:
            br = Fence(b,self)

            self.grid.place_agent(br,(h[0],h[1]))
            b += 1

    def step(self):
        '''Advance the model by one step.'''
        # Add new ants into the internal area ont he boundary

        for xy in self.neigh_bound:

            # Add with probability internal rate and if the cell is empty
            if self.random.uniform(0, 1) < self.internalrate and self.grid.is_cell_empty(xy) == True:

                a = Ant(self.ant_id, self)

                self.schedule.add(a)
                self.grid.place_agent(a,xy)

                self.ant_id += 1


        # Move the ants
        self.schedule.step()
        self.datacollector.collect(self)

        # Remove all ants on bounary

        for (agents, i, j) in self.grid.coord_iter():
            if (i,j) in self.neigh_bound and type(agents) is Ant:

                self.grid.remove_agent(agents)
                self.schedule.remove(agents)

        data_tau.append(self.mean_tau_ant)
        data_sigma.append(np.sqrt(self.sigma))
        data_sigmastar.append(self.sigmastar)

        if len(data_sigmastar) > 20:
            if abs(data_sigmastar[-2] - data_sigmastar[-1]) < 0.0000001 or len(data_sigmastar) == 2000:
                try:
                    # TAU
                    with open("results/m1_tau_5.pkl", 'rb') as f:
                        tau_old = pickle.load(f)
                        tau_old[int(len(tau_old)+1)] = data_tau
                        f.close()
                    pickle.dump(tau_old, open("results/m1_tau_5.pkl", 'wb'))

                except:
                    pickle.dump({1:data_tau}, open("results/m1_tau_5.pkl", 'wb'))

                try:
                    # SIGMA
                    with open("results/m1_sigma_5.pkl", 'rb') as f:
                        sigma_old = pickle.load(f)
                        sigma_old[int(len(sigma_old)+1)] = data_sigma
                        f.close()
                    pickle.dump(sigma_old, open("results/m1_sigma_5.pkl", 'wb'))

                except:
                    pickle.dump({1:data_sigma}, open("results/m1_sigma_5.pkl", 'wb'))

                try:
                    # SIGMASTAR
                    with open("results/m1_sigmastar_5.pkl", 'rb') as f:
                        sigmastar_old = pickle.load(f)
                        sigmastar_old[int(len(sigmastar_old)+1)] = data_sigmastar
                        f.close()
                    pickle.dump(sigmastar_old, open("results/m1_sigmastar_5.pkl", 'wb'))

                except:
                    pickle.dump({1:data_sigmastar}, open("results/m1_sigmastar_5.pkl", 'wb'))

                try:
                    # MATRIX
                    with open("results/m1_matrix_5.pkl", 'rb') as f:
                        matrix_old = pickle.load(f)
                        matrix_old[int(len(matrix_old)+1)] = self.tau
                        f.close()
                    pickle.dump(matrix_old, open("results/m1_matrix_5.pkl", 'wb'))

                except:
                    pickle.dump({1:self.tau}, open("results/m1_matrix_5.pkl", 'wb'))
                print("_______________________________________________________________________")
                print("DONE")
                self.running = False

        # with open("tau2_new.txt", "a") as myfile:
        #     myfile.write(str(self.mean_tau_ant) + '\n')
        # with open("sigma2_new.txt", "a") as myfile:
        #     myfile.write(str(np.sqrt(self.sigma)) + '\n')
        # with open("datasigmastar2_new.txt","a") as myfile:
        #     myfile.write(str(self.sigmastar) + "\n")

    def get_total_ants_number(self):
        total_ants=0
        for (agents, _, _) in self.grid.coord_iter():
            if type(agents) is Ant:
                total_ants += 1
        return total_ants

    def evaluation1(self):

        ##creat a empty grid to store currently information
        total_ants = np.zeros((WIDTH,HEIGHT))

        ## count the number of currently information
        for (agents, i, j) in self.grid.coord_iter():

            if type(agents) is Ant:
                total_ants[i][j] = 1
            else:
                total_ants[i][j] = 0

        ##update the tau
        self.tau = self.tau + total_ants

        ##calcualte the mean tau
        self.mean_tau_ant = self.tau.sum()/((WIDTH-2)**2)

        return self.mean_tau_ant

    def evaluation2(self):

        ## we need to minus the mean tau so we need to ensure the result of boundary is zero
        ## so we let the bounday equal mean_tau_ant in this way the (tau-mean_tau_ant) is zero of boundary
        for site in self.bound_vals:
            self.tau[site[0]][site[1]] = self.mean_tau_ant

        ## calculate the sigmaa
        self.sigma = ((self.tau-self.mean_tau_ant)**2).sum()/((WIDTH-2)**2)

        ## rechange the boundaryy
        for site in self.bound_vals:
            self.tau[site[0]][site[1]] = 0

        return np.sqrt(self.sigma)

    def evaluation3(self):
        ## calculate the sigmastar
        self.sigmastar = np.sqrt(self.sigma) / self.mean_tau_ant

        return self.sigmastar
