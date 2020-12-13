import math
import os
from datetime import datetime
import time
import multiprocessing as mp
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


    def get_DTW_distance_connected_component(self, sample_index, sample_2_index):
        """
        :param sample_index: the dataframe representing one time series of a handwriting sample
        :param sample_2_index: the dataframe representing one time series of another handwriting sample
        :return: the DTW distance
        """
        s1 = self.considered_time_series[sample_index].to_numpy()
        s2 = self.considered_time_series[sample_2_index].to_numpy()
        # print( self.considered_time_series[sample_index])
        # print(self.considered_time_series[sample_2_index])
        # print(type(s1), type(s2))
        # print(s1)
        # print(s2)
        # n = s1.shape[0]
        # m = s2.shape[0]
        n = s1.size
        m = s2.size
        matrix = numpy.empty((n, m))
        matrix[:] = numpy.inf
        matrix[0][0] = 0
        for i in range(0, n):
            point1 = s1[i][0]
            # print(point1, type(point1))
            for j in range(0, m):
                point2 = s2[j][0]
                # print(point2, type(point2))
                p1_p2_dist = abs(point1-point2)
                # print(p1_p2_dist, type(p1_p2_dist))
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

    def get_DTW_dist_sample_to_class(self, sample_index, filter_by_handwriting=Utils.BLOCK_LETTER, use_component = True):
        if filter_by_handwriting in self.classes[sample_index]:
            sample_dst_file_path = Utils.RES_FOLDER_PATH + Utils.TIME_DTW_DISTANCES +'_'+self.dataset_name+ self.time_stamp_last_execution+'\\'+self.considered_time_series_name+'\\'+str(sample_index)+'_'+filter_by_handwriting
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
        # print (temporary_series,'\n')
        self.get_time_stamp_last_execution()
        test_directory_path = Utils.RES_FOLDER_PATH + Utils.TIME_DTW_DISTANCES +'_'+self.dataset_name+ self.time_stamp_last_execution
        os.mkdir(test_directory_path)
        s_n = 0
        for time_series in temporary_series:
            start = time.time()
            print('START MAPPING SERIES NUMBER: ', s_n)
            self.set_considered_time_series(self.filter_by_time(time_series), self.filter_by_componet(time_series))
            print (type(self.considered_time_series), self.considered_time_series)
            self.set_considered_time_series_name(temporary_series_names[s_n])
            os.mkdir(test_directory_path+'\\'+self.considered_time_series_name)
            with mp.Pool(4) as p:
                lista_totali = list(range(len(self.classes)))
                lista_calcolati = [384, 0, 128, 256, 385, 1, 257, 129, 386, 387, 258, 130, 388, 3, 259, 389,
                                   131, 4, 390, 260, 132, 391, 392, 5, 6, 261, 262, 393, 133, 134, 7, 263,
                                   394, 135, 8, 395, 264, 396, 9, 136, 265, 397, 10, 398, 137, 266, 399, 11, 267,
                                   138, 400, 12, 268, 401, 139, 13, 402, 269, 403, 140, 14, 270, 404, 15, 405, 141,
                                   271, 406, 16, 142, 272, 407, 17, 273, 143, 408, 18, 274, 409, 144,
                                   19, 410, 275, 145, 411, 20, 276, 146, 412, 21, 413, 277, 147, 22,
                                   414, 278, 148, 23, 415, 279, 448, 24, 149, 449, 25, 280, 150, 450,
                                   26, 281, 451, 151, 27, 452, 282, 152, 28, 453, 283, 153, 454, 29, 455,
                                   284, 154, 30, 456, 285, 457, 155, 31, 286, 458, 156, 64, 459, 287,
                                   157, 460, 65, 320, 461, 158, 66, 462, 159, 321, 463, 67, 464, 192,
                                   322, 465, 68, 193, 466, 323, 69, 467, 194, 324, 468, 70, 195, 469,
                                   325, 71, 196, 470, 326, 471, 72, 197, 472, 327, 198, 73, 473, 474,
                                   328, 199, 74, 475, 200, 329, 476, 75, 201, 477, 330, 76, 202, 478,
                                   331, 479, 203, 77, 512, 332, 204, 78, 513, 205, 333, 79, 514, 206,
                                   334, 80, 515, 207, 335, 81, 208, 516, 82, 336, 209, 517, 210, 83,
                                   337, 518, 211, 519, 84, 338, 212, 520, 85, 339, 213, 521, 86, 340,
                                   214, 522, 87, 215, 341, 523, 216, 88, 342, 524, 217, 525, 343, 89,
                                   218, 526, 344, 90, 527, 219, 345, 528, 91, 220, 529, 346, 221, 530,
                                   92, 531, 347, 222, 93, 532, 223, 348, 533, 94, 534, 640, 349, 535,
                                   95, 641, 536, 350, 768, 537, 642, 351, 538, 769, 643, 539, 770, 896,
                                   644, 540, 897, 771, 645, 898, 541, 772, 899, 542, 646, 900, 773, 543,
                                   901]
                print(p.map(self.get_DTW_dist_sample_to_class, [x for x in lista_totali  if x not in lista_calcolati]))
                # print(p.map(self.get_DTW_dist_sample_to_class, [x for x in [132, 529, 916]]))
            finish = time.time()
            print('FINISH MAPPING SERIES NUMBER: ', s_n, ' in ', finish-start, ' seconds')

if __name__ == '__main__':
    timeAnalizer = TimeVariabilityAnalyzer(Utils.DATASET_NAME,  Utils.BLOCK_LETTER)
    timeAnalizer.make_DTW_distances_tables([timeAnalizer.get_samples('movementPoints')], ['movementPoints_filtered_by_time'])
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
