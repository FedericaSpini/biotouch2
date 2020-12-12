import math
from datetime import datetime

import numpy

from src import Utils
from src.DataManager import AnonymousDataManager
from src.TimeSeriesManager import TimeSeriesManager


class TimeVariabilityAnalyzer:

    def __init__(self, dataset_name, handwriting, update_data=False):
        self.dataset_name = dataset_name
        self.handwriting = handwriting
        self.time_series_manager = TimeSeriesManager(dataset_name, update_data)

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

#TODO IL CONFRONTO PER LA DISTANZA E' BIDIMENSIONALE...NON DEVE ESSERE COSI'! + se va aggioungi anche il metodo per le CC da DTWDistanceFinder
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
                # p1_p2_dist = distance.euclidean(point1, point2)
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

    def filter_by_componet(self, time_series):
        filtered_time_series = []
        for dt_frame in time_series:
            filtered_time_series.append(dt_frame[[Utils.COMPONENT]])
        return filtered_time_series

    def filter_by_time(self, time_series):
        filtered_time_series = []
        for dt_frame in time_series:
            filtered_time_series.append(dt_frame[[Utils.TIME]])
        return filtered_time_series

    #TODO: AGGIUNGI, DA DTWDISTANCEFINDER, I METODI GET_DTW_DIST_SAMPLE_TO_CLASS + make_DTW_distances_tables

    # def time_analysis(self):
    #     time_serie_mv_points =self.datamanager.data_frames['movementPoints']['time']
    #     cc_serie_mv_points = self.datamanager.data_frames['movementPoints']['component']
    #     print(time_serie_mv_points)
    #     print(cc_serie_mv_points)
    #
    #     cc_points_number = []
    #     touch_down_points_time = []
    #     touch_up_points_time = []
    #     relative_touch_down_points_time = []
    #     relative_touch_up_points_time = []
    #     old_v = 0
    #     old_cc = 0
    #     cc_point_counter = 0
    #
    #     for n in range(cc_serie_mv_points.size):
    #         print(cc_serie_mv_points[n])
    #         # if not (cc_serie_mv_points[n] == old_cc):
    #         #     print(cc_serie_mv_points[n], old_cc)
    #         #     cc_points_number.append(cc_point_counter)
    #         #     cc_point_counter = 0
    #         # else:
    #         #     cc_point_counter += 1
    #         # old_cc = cc_serie_mv_points[n]
    #
    #         # print(time_serie_mv_points[n], time_serie_mv_points[n]-old_v)
    #         # print()
    #         # old_v = time_serie_mv_points[n]
    #     print(cc_points_number)




if __name__ == '__main__':
    timeAnalizer = TimeVariabilityAnalyzer(Utils.MINI_DATASET_NAME,  Utils.BLOCK_LETTER)

    # print(d.dataset_name, "\n\n")
    # print(type(d.data_frames['wordid_userid_map']),d.data_frames.keys(), "\n\n")

    # print(d.data_frames['userid_userdata_map'][4], d.data_frames['userid_userdata_map'][4])
    # for k in d.data_frames.keys():
    #     print(k)
        # print('\n')
        # print(d.data_frames[k])
    # print(d.data_frames['movementPoints'][['x', 'y', 'time']])
    # print('\n')
    # print(d.data_frames.keys())

    # a = Utils.get_wordidfrom_wordnumber_name_surname(d[Utils.WORDID_USERID], d[Utils.USERID_USERDATA], "Rita", "Battilocchi" , Utils.BLOCK_LETTER, 31)
    # print(Utils.get_infos(d[Utils.WORDID_USERID], d[Utils.USERID_USERDATA], a))
    # d._generate_example_charts()
