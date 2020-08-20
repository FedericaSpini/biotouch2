# -*- coding: utf-8 -*-
import json
import random
import pandas
import os

import numpy
import src.Chronometer as Chronom
import src.Utils as Utils
from src.DTWDistMatrix import DTWDistMatrix
from src.DTWDistMatrixManager import DTWDistMatrixManager
from src.DataManager import AnonymousDataManager
from src.TimeSeriesManager import TimeSeriesManager


class DTWClassifier:

    def __init__(self, dataset_name,  dtwstMatrix, update_data = False):
        self.dm = AnonymousDataManager(dataset_name, update_data)
        self.time_series_manager = TimeSeriesManager(dataset_name, update_data)
        self.dtwstMatrix = dtwstMatrix
        self.correct_classification = self.get_correct_classification()

    def get_correct_classification(self):
        cc = {}
        classes = self.time_series_manager.get_classes()
        for label in self.dtwstMatrix.get_label_set():
            cc[label] = classes[label]
        return cc

    def classify_by_min_dist(self):
        classification = {}
        for s1 in self.dtwstMatrix.get_label_set():
            min_distance = numpy.inf
            correct_class = ''
            for s2 in self.dtwstMatrix.get_label_set():
                if (s1 != s2):
                    if (s1, s2) in self.dtwstMatrix.get_couples_to_dist():
                        distance = self.dtwstMatrix.get_dist(s1, s2)[0]
                        if(distance<min_distance):
                            min_distance = distance
                            correct_class = self.correct_classification[s2]
                    else:
                        if (s1 != s2):
                            print('\nTHE COUPLE: ', s1, ' - ',s2,' IS NOT IN THE DATASET!!!\n')
            classification[s1] = correct_class
        return classification

    def classify_by_avg_dist(self):
        classification = {}
        for s1 in self.dtwstMatrix.get_label_set():
            classes_sum_value = {}
            classes_sample_number = {}
            classes_avg_values = {}
            for s2 in self.dtwstMatrix.get_label_set():
                if s1 != s2:
                    if (s1, s2) in self.dtwstMatrix.get_couples_to_dist():
                        distance = self.dtwstMatrix.get_dist(s1, s2)[0]
                        s2_class = self.correct_classification[s2]
                        if s2_class in classes_sample_number:
                            classes_sample_number[self.correct_classification[s2]]+=1
                            classes_sum_value[self.correct_classification[s2]] += distance
                        else:
                            classes_sample_number[self.correct_classification[s2]] = 1
                            classes_sum_value[self.correct_classification[s2]] = distance
                    else:
                        if s1 != s2:
                            print('\nTHE COUPLE: ', s1, ' - ',s2,' IS NOT IN THE DATASET!!!\n')
            for k in classes_sum_value:
                classes_avg_values[k] = classes_sum_value[k]/classes_sample_number[k]
            min_avg_dist = numpy.inf
            correct_class = ''
            for k in classes_avg_values:
                if classes_avg_values[k] < min_avg_dist:
                    min_avg_dist = classes_avg_values[k]
                    correct_class = k
            classification[s1] = correct_class
        return classification

    def evaluate_classification(self, classification):
        classes_correctness = {}
        for k in self.get_correct_classification():
            k_correctness = classification[k] == self.get_correct_classification()[k]
            if self.get_correct_classification()[k] in classes_correctness:
                if k_correctness:
                    classes_correctness[self.get_correct_classification()[k]][0] += 1
                else:
                    classes_correctness[self.get_correct_classification()[k]][1] += 1
            else:
                if k_correctness:
                    classes_correctness[self.get_correct_classification()[k]] = [1, 0]
                else:
                    classes_correctness[self.get_correct_classification()[k]] = [0, 1]
        print(classes_correctness)
        return classes_correctness

if __name__ == '__main__':
    classifier = DTWClassifier(Utils.DATASET_NAME, DTWDistMatrixManager(Utils.DATASET_NAME).get_matrix('movementPoints_filtered_by_x_y'))
    simple_min_dist_classification = classifier.classify_by_min_dist()
    simple_avg_dist_classification = classifier.classify_by_avg_dist()
    print(classifier.get_correct_classification())
    # print(simple_min_dist_classification)
    # print(simple_avg_dist_classification)

    classifier.evaluate_classification(simple_min_dist_classification)
    classifier.evaluate_classification(simple_avg_dist_classification)