import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os

## sEMG Plot Function from text file

class sEMGData:

    def __init__(self, path):
        self.path = path

        # file name
        f = open(path, 'r')
        basename = os.path.basename(f.name)

        if 'mar' in basename:
            self.output = 'mar'
        elif 'pie' in basename:
            self.output = 'pie'
        elif 'sen' in basename:
            self.output = 'sen'

        # invalid_raise - sk
        # numpy array
        self.array = np.genfromtxt(self.path, skip_header=7, invalid_raise=False)
        # DataFrame
        self.dataframe = pd.DataFrame(self.array, columns=['RF','BF','VM','ST','FX'])

    def return_array(self):

        return self.array

    def return_df(self):

        return self.dataframe

    def return_output(self):
        return self.output

    def plot_RF(self):

        plt.plot(self.dataframe['RF'])
        plt.title('RF - rectus femoris')
        plt.show()

    def plot_BF(self):

        plt.plot(self.dataframe['BF'])
        plt.title('BF - biceps femoris')
        plt.show()

    def plot_VM(self):

        plt.plot(self.dataframe['VM'])
        plt.title('VM - vastus internus')
        plt.show()

    def plot_ST(self):

        plt.plot(self.dataframe['ST'])
        plt.title('ST - semitendinosus')
        plt.show()

    def plot_FX(self):

        plt.plot(self.dataframe['FX'])
        plt.title('FX - knee flexion ()')
        plt.show()

    def sEMG_plot(self):

        # Plot individual sEMG data

        plt.plot(self.dataframe['RF'])
        plt.title('RF - rectus femoris')
        plt.show()

        plt.plot(self.dataframe['BF'])
        plt.title('BF - biceps femoris')
        plt.show()

        plt.plot(self.dataframe['VM'])
        plt.title('VM - vastus internus')
        plt.show()

        plt.plot(self.dataframe['ST'])
        plt.title('ST - semitendinosus')
        plt.show()

        plt.plot(self.dataframe['FX'])
        plt.title('FX - knee flexion ()')
        plt.show()

# Preprocessing Function

def preprocess(folder):

    input_array = []
    classes = []

    # iterates through folder
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            # instantiates each sEMG data
            sEMG = sEMGData(folder + filename)

            input_array.append(sEMG.return_array())
            classes.append(sEMG.return_output())

    # finds the max length of a sEMG sequence
    max_len = max([len(i) for i in input_array])

    for i in input_array:
        if (len(i) == max_len):
            largest_array = i

    # list of padded sEMG numpy arrays
    padded_array = []

    # loop through input_array and append into new padded array
    for i in input_array:
        # zero numpy array
        zero_array = np.zeros(largest_array.shape)
        zero_array[:i.shape[0],:i.shape[1]] = i

        padded_array.append(zero_array)

    # convert to 3d numpy array
    padded_array = np.dstack(padded_array)

    # transpose array
    padded_array = padded_array.transpose(2,0,1)

    return padded_array, classes
    
# Plotting function - based on muscle group

def plot_dataset(input_array, class_array, muscle_group):
    for i in range(0, len(input_array)):

        df = pd.DataFrame(input_array[i], columns=['RF','BF','VM','ST','FX'])

        # show plot of each rectus femoris sEMG data
        plt.plot(df[muscle_group])
        plt.title(muscle_group + ' - ' + class_array[i])
        plt.show()
