# -*- coding: utf-8 -*-
import json
import random
import pandas
import os

import src.Chronometer as Chronom
import src.Utils as Utils

class DTWDistMatrix:

    def __init__(self, dir):
        self.couples_to_dist = {}
        self._load_data(dir)
        # print(self.couples_to_dist)


    def _load_data(self, dir):
        for sample_row in (os.listdir(dir)):
            sample_number = sample_row.split('_')[0]
            with open(dir + '\\' + sample_row, 'r') as fp:
                line = fp.readline()
                while line:
                    row = line.strip().split(',')
                    line = fp.readline()
                    self.couples_to_dist[(int(sample_number), int(row[0]))] = (float(row[1]),  int(row[2]),  int(row[3]))
                    self.couples_to_dist[(int(row[0]), int(sample_number))] = (float(row[1]),  int(row[2]),  int(row[3]))

    def get_dist (self, s1, s2):
        dist = self.couples_to_dist[(s1,s2)]
        print(dist)
        print(dist[0])
        print(dist[1])
        print(dist[2])
        return dist





if __name__ == '__main__':
    matrix = DTWDistMatrix(Utils.DATASET_NAME)