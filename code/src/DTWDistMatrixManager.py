# -*- coding: utf-8 -*-
import json
import random
import pandas
import os

import src.Chronometer as Chronom
import src.Utils as Utils
from src.DTWDistMatrix import DTWDistMatrix


class DTWDistMatrixManager:

    def __init__(self, dataset_name, update_data=False, res_path = Utils.RES_FOLDER_PATH+Utils.FINAL_DTW_DISTANCES):
        self.matrixes = {}
        self.res_path = res_path
        self._load_data()


    def _load_data(self):
        for ts_name in os.listdir(self.res_path):
            mat = DTWDistMatrix(self.res_path+'\\'+ts_name)
            self.matrixes[ts_name] = mat

    def get_all_matrixes(self):
        return self.matrixes

    def get_matrix(self, k):
        return self.matrixes[k]


if __name__ == '__main__':
    matrix2 = DTWDistMatrixManager(Utils.DATASET_NAME, res_path=Utils.RES_FOLDER_PATH+Utils.FINAL_DTW_DISTANCES_TIME)
    # print(matrix2.matrixes, matrix2.res_path)
    # print(matrix2.get_matrix('movementPoints_filtered_by_time').get_dist(1,2))
    # print(matrix2.get_matrix('movementPoints_filtered_by_time').get_dist(2,1))
    # print(matrix2.get_matrix('movementPoints_filtered_by_time').get_dist(1, 1999))