import os
from datetime import datetime

import numpy
import multiprocessing as mp
import time
import math

import src.TimeSeriesManager as sm
from src import Utils

LEARNING_FROM = Utils.TIMED_POINTS_SERIES_TYPE
# MOVEMENT_WEIGHT = 0.75


class DTWDistanceFinder:

    def __init__(self, dataset_name, handwriting, update_data=False):
        self.dataset_name = dataset_name
        self.handwriting = handwriting

        self.time_series_manager = sm.TimeSeriesManager(dataset_name, update_data)

        self.series = self.time_series_manager.get_series()
        self.classes = self.time_series_manager.get_samples_id()

        self.considered_time_series = None
        self.considered_time_series_components = None
        self.considered_time_series_name = ''
        self.time_stamp_last_execution = ""

    def get_time_stamp_last_execution(self):
        self.time_stamp_last_execution = datetime.now().strftime('_%d_%b_%Y.%H.%M.%S')

    def set_considered_time_series (self, time_series, components):
        """
        :param time_series: a list of dataframes which represent the points of a time series,
                            and set it as the series on which apply DTW
        """
        self.considered_time_series = time_series
        self.considered_time_series_components = components

    def set_considered_time_series_name (self, time_series_name):
        """
        :param time_series: a list of dataframes which represent the points of a time series,
                            and set it as the series on which apply DTW
        """
        self.considered_time_series_name = time_series_name

    def get_samples(self, series_name):
        return self.series[series_name]

    def get_DTW_distance(self, sample_index, sample_2_index):
        """
        :param s1: the dataframe representing one time series of a handwriting sample
        :param s2: the dataframe representing one time series of another handwriting sample
        :return: the DTW distance
        """
        s1 = self.considered_time_series[sample_index]
        s2 = self.considered_time_series[sample_2_index]
        n = s1.shape[0]
        m = s2.shape[0]
        matrix = numpy.empty((n, m))
        matrix[:] = numpy.inf
        matrix[0][0] = 0
        for i in range(0, n):
            point1 = s1.iloc[i].to_numpy()
            for j in range(0, m):
                point2 = s2.iloc[j].to_numpy()
                p1_p2_dist = math.sqrt(((point1[0]-point2[0])**2)+(point1[1]-point2[1])**2)
                if (i == 0) and (j > 0):
                    matrix[i][j] = p1_p2_dist + matrix[i][j-1]
                elif (i > 0) and (j == 0):
                    matrix[i][j] = p1_p2_dist + matrix[i-1][j]
                elif(i!=0) and (j!=0):
                    matrix[i][j] = p1_p2_dist + min(matrix[i - 1][j - 1],
                                                            min(matrix[i - 1][j],
                                                                matrix[i][j - 1]))
        return matrix[n-1][m-1]

    def get_DTW_distance_connected_component(self, sample_index, sample_2_index):
        """
        :param s1: the dataframe representing one time series of a handwriting sample
        :param s2: the dataframe representing one time series of another handwriting sample
        :return: the DTW distance
        """
        s1 = self.considered_time_series[sample_index]
        s2 = self.considered_time_series[sample_2_index]
        n = s1.shape[0]
        m = s2.shape[0]
        matrix = numpy.empty((n, m))
        matrix[:] = numpy.inf
        matrix[0][0] = 0
        for i in range(0, n):
            point1 = s1.iloc[i].to_numpy()
            for j in range(0, m):
                point2 = s2.iloc[j].to_numpy()
                p1_p2_dist = math.sqrt(((point1[0]-point2[0])**2)+(point1[1]-point2[1])**2)
                if (i == 0) and (j > 0):
                    matrix[i][j] = p1_p2_dist + matrix[i][j-1]
                elif (i > 0) and (j == 0):
                    matrix[i][j] = p1_p2_dist + matrix[i-1][j]
                elif (i != 0) and (j != 0):
                    matrix[i][j] = p1_p2_dist + min(matrix[i - 1][j - 1],
                                                            min(matrix[i - 1][j],
                                                                matrix[i][j - 1]))
        different_component_count = 0
        total_couple_count = 1
        last_cell_x, last_cell_y = n-1, m-1
        while (last_cell_x != 0) or (last_cell_y != 0):
            p1_component, p2_component = self.considered_time_series_components[sample_index].iloc[last_cell_x].to_numpy(), self.considered_time_series_components[sample_2_index].iloc[last_cell_y].to_numpy()
            if p1_component != p2_component:
                different_component_count += 1
            sx, up, diag = matrix[last_cell_x-1][last_cell_y], matrix[last_cell_x][last_cell_y-1], matrix[last_cell_x-1][last_cell_y-1]
            next_val = min(sx, up, diag)
            total_couple_count += 1
            if sx == next_val:
                last_cell_x -= 1
            elif up == next_val:
                last_cell_y -= 1
            else:
                last_cell_y -= 1
                last_cell_x -= 1
        return matrix[n-1][m-1], different_component_count, total_couple_count

    def filter_time_series_by_x_y(self, time_series):
        filtered_time_series = []
        for dt_frame in time_series:
            filtered_time_series.append(dt_frame[[Utils.X, Utils.Y]])
        return filtered_time_series

    def filter_by_componet(self, time_series):
        filtered_time_series = []
        for dt_frame in time_series:
            filtered_time_series.append(dt_frame[[Utils.COMPONENT]])
        return filtered_time_series


    def get_DTW_dist_sample_to_class(self, sample_index, filter_by_handwriting=Utils.BLOCK_LETTER, use_component = True):
        if filter_by_handwriting in self.classes[sample_index]:
            sample_dst_file_path = Utils.RES_FOLDER_PATH + Utils.RES_DTW_DISTANCES +'_'+self.dataset_name+ self.time_stamp_last_execution+'\\'+self.considered_time_series_name+'\\'+str(sample_index)+'_'+filter_by_handwriting
            print('\nSTART TO FIND DISTANCES WITH CLASSES FOR THE ', sample_index, ' SAMPLE')
            total_class_set = set(self.classes)
            class_set = set()
            for c in total_class_set:
                if filter_by_handwriting in c:
                    class_set.add(c)

            outF = open(sample_dst_file_path,'w')
            for c in class_set:
                sum = 0
                min = numpy.inf
                indices = [i for i, x in enumerate(self.classes) if x == c]
                for ind in indices:
                    if (ind > sample_index):
                        if not use_component:
                            val = self.get_DTW_distance(sample_index, ind)
                            outF.write(str(ind)+','+str(val)+'\n')
                        else:
                            val, component_dist, path_length = self.get_DTW_distance_connected_component(sample_index, ind)
                            outF.write(str(ind)+','+str(val)+','+str(component_dist)+','+str(path_length)+'\n')
                        sum += val
                        if val < min:
                            min = val
            outF.close()
            print ('\nDISTANCES FOR THE ', sample_index, ' SAMPLE have been found!')
        return 0


    def make_DTW_distances_tables(self, temporary_series, temporary_series_names):
        """
        :param self:
        :param temporary_series: a list of the temporary series to consider
        """
        self.get_time_stamp_last_execution()
        test_directory_path = Utils.RES_FOLDER_PATH + Utils.RES_DTW_DISTANCES +'_'+self.dataset_name+ self.time_stamp_last_execution
        os.mkdir(test_directory_path)
        s_n = 0
        for time_series in temporary_series:
            start = time.time()
            print('START MAPPING SERIES NUMBER: ', s_n)
            self.set_considered_time_series(a.filter_time_series_by_x_y(time_series), a.filter_by_componet(time_series))
            self.set_considered_time_series_name(temporary_series_names[s_n])
            os.mkdir(test_directory_path+'\\'+self.considered_time_series_name)
            with mp.Pool(mp.cpu_count()) as p:
                lista_totali = list(range(len(self.classes)))
                lista_calcolati =[]
                print(p.map(self.get_DTW_dist_sample_to_class, [x for x in lista_totali]))
                # print(p.map(self.get_DTW_dist_sample_to_class, [x for x in [132, 529, 916]]))
            finish = time.time()
            print('FINISH MAPPING SERIES NUMBER: ', s_n, ' in ', finish-start, ' seconds')


if __name__ == '__main__':
    a = DTWDistanceFinder(Utils.DATASET_NAME, Utils.ITALIC)
    a.make_DTW_distances_tables([a.get_samples('movementPoints')], ['movementPoints_filtered_by_x_y'])