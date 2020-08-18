# -*- coding: utf-8 -*-
import json
import random
import pandas
import os

import src.Chronometer as Chronom
import src.Utils as Utils
from src.DTWDistMatrix import DTWDistMatrix


class DTWDistMatrixManager:

    def __init__(self, dataset_name, update_data=False):
        self.matrixes = {}
        self._load_data()


    def _load_data(self):
        for ts_name in os.listdir(Utils.RES_FOLDER_PATH + Utils.FINAL_DTW_DISTANCES):
            mat = DTWDistMatrix(Utils.RES_FOLDER_PATH + Utils.FINAL_DTW_DISTANCES+'\\'+ts_name)
            self.matrixes[ts_name] = mat

    def get_all_matrixes(self):
        return self.matrixes

    def get_matrix(self, k):
        return self.matrixes[k]


if __name__ == '__main__':
    matrix = DTWDistMatrixManager(Utils.DATASET_NAME)
    print(matrix.get_matrix('movementPoints_filtered_by_x_y').get_dist(1,2))
    print(matrix.get_matrix('movementPoints_filtered_by_x_y').get_dist(2,1))
    print(matrix.get_matrix('movementPoints_filtered_by_x_y').get_dist(1, 1999))