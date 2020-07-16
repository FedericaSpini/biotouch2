from datetime import datetime

import sklearn
import sklearn.model_selection
import pandas
import numpy

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

    def getDTWDistance(self, s1, s2):
        n = len(s1)
        m = len(s2)
        matrix = numpy.empty((n+1, m+1))
        matrix[:] = numpy.nan
        matrix[0][0] = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                if (i == 0) and (j > 0):
                    matrix[i][j] = abs(s1[i]-s2[j]) + matrix[i][j-1]
                elif (i > 0) and (j == 0):
                    matrix[i][j] = abs(s1[i] - s2[j]) + matrix[i-1][j]
                else:
                    matrix[i][j] = abs(s1[i] - s2[j]) + min(matrix[i - 1][j - 1],
                                                            min(matrix[i - 1][j],
                                                                matrix[i][j - 1]))
        return matrix[n+1][m+1]


if __name__ == '__main__':
    a = DTWWordClassifier(Utils.MINI_DATASET_NAME, Utils.ITALIC)
    # print(a)