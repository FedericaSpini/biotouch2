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
        self.label_set = set()
        self._load_data(dir)
        # print(self.couples_to_dist)


    def _load_data(self, dir):
        for sample_row in (os.listdir(dir)):
            sample_number = int(sample_row.split('_')[0])
            with open(dir + '\\' + sample_row, 'r') as fp:
                line = fp.readline()
                while line:
                    row = line.strip().split(',')
                    line = fp.readline()
                    self.couples_to_dist[(sample_number, int(row[0]))] = (float(row[1]),  int(row[2]),  int(row[3]))
                    self.couples_to_dist[(int(row[0]), sample_number)] = (float(row[1]),  int(row[2]),  int(row[3]))
                    self.label_set.add(sample_number)
                    self.label_set.add(int(row[0]))

    def get_dist (self, s1, s2):
        dist = self.couples_to_dist[(s1,s2)]
        # print(dist)
        return dist

    def get_label_set(self):
        return self.label_set

    def get_couples_to_dist(self):
        return self.couples_to_dist

    # def get_smaller_dist (self, s):
    #     for





if __name__ == '__main__':
    matrix = DTWDistMatrix(Utils.DATASET_NAME)