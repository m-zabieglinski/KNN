import numpy as np
import random as rndm
from matplotlib import pyplot as plt
from math import e

#measures distance between vectors v1 and v2, by default using the eucledian distance definition
#if the vectors are of different length, it calculates the distance only up to their common indexes
#eg. v1 = [0, 1, 1] and v2 = [1, 1] -> the euclidian distance is ((0-1)**2 + (1-1)**2)**0.5 = 1
def distance(v1, v2, dist = "euclid"):
    if dist == "euclid":
        distance = 0
        if len(v1) > len(v2):
            for index in range(len(v2)):
                distance += (v2[index] - v1[index]) ** 2 #probably can be done faster using list comprehension
        else:
            for index in range(len(v1)):
                distance += (v2[index] - v1[index]) ** 2 #probably can be done faster using list comprehension
        distance = distance ** 0.5
    return distance

#normalizes the input dataset to [0,1] (like the network's)
def normalize(dataset):

    def norm(x, A, B):  #maps given number in range [A,B] onto a number in range [0,1]
        return (x - A) / (B - A)
    
    row_length = len(min(dataset, key = len))
    
    max_val = [] #finding max and min values for each variable
    min_val = []
    for i in range(row_length):    
        max_val.append(max(dataset, key = lambda x: x[i])[i])
        min_val.append(min(dataset, key = lambda x: x[i])[i])
        
    norm_data = [] #normalizing the dataset row by row
    for row in dataset: 
        norm_data.append([norm(row[i], min_val[i], max_val[i]) for i in range(row_length)])
    return norm_data

class KN:
    
    def __init__(self, variable_number = 2, size = 2):
        self.N = size #network size NxN
        self.vector_length = variable_number #how many variables are in each node / input vector
        self.network = np.empty([size, size], dtype = object) #initializing first network with no weights
        
    def create(self, dataset = None):
        self.network
        if dataset == None:
            with np.nditer(self.network, op_flags = ["writeonly"], flags = ["refs_ok"]) as it:
              for neuron in it: #creates random weights between 0 and 1 for each variable in each node
                  neuron[...] = [rndm.uniform(0,1) for var in range(self.vector_length)]
      
    def train(self, dataset, epochs = 0.5, step = 1):
        
        def conv(x): #function that describes how the step changes in each epoch (converges to 0)
            return e ** (-0.5 * x)
        data = normalize(dataset)
        for epoch in range(epochs):
            print(f"\rTraining on the dataset, epoch {epoch+1} of {epochs}", sep=' ', end='', flush = True)
            for item in data:
                max_inv_dist = 0 #operating on maximum inverse distance instead of minimum distance since the former has a limit at 0
                for row in range(0, self.N): #this nested loop could probably be done faster with np.nditer
                    for col in range(0, self.N):
                        dist = distance(item, self.network[row][col])
                        if 1 / dist > max_inv_dist: #checking if the neuron is a better match
                            max_inv_dist = 1 / dist
                            winner = [row, col] # assigning the winner
                #modifying the winner and its neighbourhood here
                self.network[winner[0]][winner[1]] = np.add(self.network[winner[0]][winner[1]],
                                                step * conv(epoch) * np.subtract(item, self.network[winner[0]][winner[1]]))
                nhood_rows = [index for index in range(winner[0]-1, winner[0]+2) if (index <= self.N-1 and index >= 0)]
                nhood_cols = [index for index in range(winner[1]-1, winner[1]+2) if (index <= self.N-1 and index >= 0)]
                for row in nhood_rows: #col and row in case something breaks with the outer loop with col and row
                    for col in nhood_cols:
                        self.network[row][col] = np.add(self.network[row][col],
                                                step * 0.5 * conv(epoch) * np.subtract(item, self.network[row][col]))
    def categorize(self, dataset):
        data = normalize(dataset) #from here onwards, repeating the procedure for train, but just assigning the map nodes to observations
        mapped_data = []
        for item in data:
            max_inv_dist = 0
            for row in range(0, self.N):
                for col in range(0, self.N):
                    dist = distance(item, self.network[row][col])
                    if 1 / dist > max_inv_dist:
                        max_inv_dist = 1 / dist
                        winner = [row, col]
            mapped_data.append(winner)
        return mapped_data
            
    def net(self):
        return self.network
    
    def show(self):
        print(self.network)
        
    def showmap(self, dataset = None):
            if np.array(dataset).shape[1] >= 2:
                data = normalize(dataset)
                plt.scatter(x = [item[0] for item in data], y = [item[1] for item in data], c = "blue", alpha = 0.5)
            if self.vector_length == 2:
                neuron_x = [] #1st axis for neuron locations
                neuron_y = [] #2nd axis for neuron locations
                for row in range(0, self.N): #this can also probably be done faster with np.nditer
                    for col in range(0, self.N):
                        neuron_x.append(self.network[row][col][0])
                        neuron_y.append(self.network[row][col][1])
                plt.scatter(x = neuron_x, y = neuron_y, c = "red", alpha = 0.5)
            else:
                print("I only draw the map for networks in 2D space")
        
    
            

            
            