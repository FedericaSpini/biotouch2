from datetime import datetime

import sklearn
import sklearn.model_selection
import pandas
import numpy
import multiprocessing as mp
from scipy.spatial import distance
import time

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

        self.considered_time_series = None

    def set_considered_time_series (self, time_series):
        self.considered_time_series = time_series

    def get_samples(self, series_name):
        return self.series[series_name]

    def get_DTW_distance(self, s1, s2):
        """
        :param s1: the dataframe representing one time series of a handwriting sample
        :param s2: the dataframe representing one time series of another handwriting sample
        :return: the DTW distance
        """
        n = s1.shape[0]
        m = s2.shape[0]
        matrix = numpy.empty((n, m))
        matrix[:] = numpy.inf
        matrix[0][0] = 0
        # print(matrix)
        for i in range(1, n):
            point1 = s1.iloc[i].to_numpy()
            for j in range(1, m):
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

    def get_DTW_distance2(self, s1, s2):
        """
        :param s1: the dataframe representing one time series of a handwriting sample
        :param s2: the dataframe representing one time series of another handwriting sample
        :return: the DTW distance
        """
        # print('START')
        n = s1.shape[0]
        m = s2.shape[0]
        matrix = numpy.empty((n, m))
        matrix[:] = numpy.inf
        matrix[0][0] = 0
        # print(matrix)
        for i in range(1, n):
            point1 = s1.iloc[i].to_numpy()
            for j in range(1, m):
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

    def filter_time_series_by_x_y(self, time_series):
        filtered_time_series = []
        for dt_frame in time_series:
            filtered_time_series.append(dt_frame[['x','y']])
        return filtered_time_series


    def get_DTW_dist_sample_to_class(self, sample_index):
        print ('\nSTART TO FIND DISTANCES WITH CLASSES FOR THE ', sample_index, ' SAMPLE')
        total_class_set = set(self.classes)
        class_set = set()
        sample = self.considered_time_series[sample_index]
        correct_class = self.classes[sample_index]
        min_distances = {}
        avg_distances = {}
        if Utils.ITALIC in correct_class:
            for c in total_class_set:
                if Utils.ITALIC in c:
                    class_set.add(c)
        if Utils.BLOCK_LETTER in correct_class:
            for c in total_class_set:
                if Utils.BLOCK_LETTER in c:
                    class_set.add(c)
        for c in class_set:
            sum = 0
            min = numpy.inf
            indices = [i for i, x in enumerate(self.classes) if x == c]
            

            # SECOND WAY----------------------------------------------------------------------
            #works, but time stays the same
            # pool = mp.Pool(mp.cpu_count())
            # with mp.Pool(mp.cpu_count()) as pool:
            #     res = [pool.apply(a.get_DTW_distance2, args=(sample, time_series[s_index])) for s_index in indices]
            #     print(res)
            # pool.close()
            # ------------------------------------------------------------------------

            # FIRST WAY----------------------------------------------------------------------
            #without any parallelization
            for ind in indices:
                if (ind != sample_index):
                    val = self.get_DTW_distance(sample, self.considered_time_series[ind])
                    sum += val
                    if val < min:
                        min = val
            avg_distances[c] = sum/len(indices)
            min_distances[c] = min
            #------------------------------------------------------------------------

        print ('\nDISTANCES FOR THE ', sample_index, ' SAMPLE: ', correct_class, '\nAVG_DISTANCES: ', avg_distances, '\nMIN_DISTANCES: ', min_distances)




if __name__ == '__main__':
    start = time.time()

    a = DTWWordClassifier(Utils.MINI_DATASET_NAME, Utils.ITALIC)
    a.set_considered_time_series(a.filter_time_series_by_x_y(a.get_samples('movementPoints')))


    with mp.Pool(mp.cpu_count()) as p:
        print(p.map(a.get_DTW_dist_sample_to_class, list(range(len(a.classes)))))



    finish = time.time()
    print(finish-start)