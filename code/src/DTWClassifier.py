from datetime import datetime

import sklearn
import sklearn.model_selection
import pandas
import numpy
from scipy.spatial import distance

import src.Constants
import random
import src.TimeSeriesManager as sm
from src import Utils

LEARNING_FROM = Utils.TIMED_POINTS_SERIES_TYPE
MOVEMENT_WEIGHT = 0.75


class DTWWordClassifier:

    def __init__(self, dataset_name, handwriting, test_size=0.3125, update_data=False, check_consistency=False, anonymous=True):
        self.dataset_name = dataset_name
        self.handwriting = handwriting

        self.time_series_manager = sm.TimeSeriesManager(dataset_name, update_data)

        self.series = self.time_series_manager.get_series()
        self.classes = self.time_series_manager.get_classes()

    def get_samples(self, series_name):
        # print('\n\n\n\n\n')
        # print(type(self.series[series_name][0]))
        # print('\n\n\n\n\n')
        return self.series[series_name]

    def get_DTW_distance(self, s1, s2):
        """
        :param s1: the dataframe representing one time series of a handwriting sample
        :param s2: the dataframe representing one time series of another handwriting sample
        :return: the DTW distance
        """
        n = s1.shape[0]
        m = s2.shape[0]
        print(n, ' - ', m)
        matrix = numpy.empty((n, m))
        matrix[:] = numpy.inf
        matrix[0][0] = 0
        print(matrix)
        for i in range(1, n):
            print('---> i VALE ', i)
            point1 = s1.iloc[i].to_numpy()
            for j in range(1, m):
                print('---> j VALE ', j)
                point2 = s2.iloc[j].to_numpy()
                if (i == 0) and (j > 0):
                    matrix[i][j] = distance.euclidean(point1, point2) + matrix[i][j-1]
                elif (i > 0) and (j == 0):
                    matrix[i][j] = distance.euclidean(point1, point2) + matrix[i-1][j]
                else:
                    matrix[i][j] = distance.euclidean(point1, point2) + min(matrix[i - 1][j - 1],
                                                            min(matrix[i - 1][j],
                                                                matrix[i][j - 1]))
        return matrix[n-1][m-1]


if __name__ == '__main__':
    a = DTWWordClassifier(Utils.MINI_DATASET_NAME, Utils.ITALIC)
    print((a.get_samples('movementPoints')[0].iloc[0].to_numpy()))
    print(type(a.get_samples('movementPoints')[0].iloc[0].to_numpy()))
    print(distance.euclidean(a.get_samples('movementPoints')[0].iloc[0].to_numpy(),
                       a.get_samples('movementPoints')[5].iloc[0].to_numpy()))
    print("\n\n\n")
    print(a.get_DTW_distance(a.get_samples('movementPoints')[35],
                             a.get_samples('movementPoints')[34]))
    # print((a.get_samples('movementPoints')[1][0]))
    # print(a)