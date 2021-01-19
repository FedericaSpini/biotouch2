# -*- coding: utf-8 -*-
import json
import random
import pandas
import os

import numpy
import src.Chronometer as Chronom
import src.Utils as Utils
from src import Constants
from src.DTWDistMatrix import DTWDistMatrix
from src.DTWDistMatrixManager import DTWDistMatrixManager
from src.DataManager import AnonymousDataManager
from src.TimeSeriesManager import TimeSeriesManager


class DTWClassifier:

    def __init__(self, dataset_name,  dtwstMatrix, update_data = False):
        """
        :param dataset_name: the name of the dataset whose data will be classified by tje DTWClassifier object
        :param dtwstMatrix: a distance matrix given such a DTWDistMatrix object
        :param update_data: a boolean wich specify if this object updates already existing data
        """
        self.dm = AnonymousDataManager(dataset_name, update_data)
        self.time_series_manager = TimeSeriesManager(dataset_name, update_data)
        self.dtwstMatrix = dtwstMatrix
        self.correct_classification = self.get_correct_classification()

    def get_correct_classification(self):
        """
        :return: a dictiory wich maps probes'ids into the correct class label
        """
        cc = {}
        classes = self.time_series_manager.get_samples_id()
        for label in self.dtwstMatrix.get_label_set():
            cc[label] = classes[label]
        return cc

    def get_classes_number(self):
        """
        :return: the total number of classes
        """
        return len(self.get_classes_set())

    def get_classes_set(self):
        """
        :return: the set containing all the classes'labels of the dataset
        """
        classes_set = set()
        cc = self.get_correct_classification()
        for k in cc:
            classes_set.add(cc[k])
        return classes_set

    def classify_by_min_dist(self):
        """
        :return: a dictionary which maps each probe id into another dictionary that is: class name -> minimum distance
                    of the probe to classes probe
        """
        classification = {}
        for s1 in self.dtwstMatrix.get_label_set():
            min_dist_dict = {}
            for s2 in self.dtwstMatrix.get_label_set():
                if s1 != s2:
                    s2_class = self.correct_classification[s2]
                    if s2_class in min_dist_dict:
                        if self.dtwstMatrix.get_dist(s1, s2)[Constants.ZERO] < min_dist_dict[s2_class]:
                            min_dist_dict[s2_class] = self.dtwstMatrix.get_dist(s1, s2)[Constants.ZERO]
                    else:
                        min_dist_dict[s2_class] = self.dtwstMatrix.get_dist(s1, s2)[Constants.ZERO]
            classification[s1] = min_dist_dict
        return classification

    def classify_by_max_dist(self):
        """
        :return: a dictionary which maps each probe id into another dictionary that is: class name -> maximum distance
                    of the probe to classes probe
        """
        classification = {}
        for s1 in self.dtwstMatrix.get_label_set():
            max_dist_dict = {}
            for s2 in self.dtwstMatrix.get_label_set():
                if s1 != s2:
                    s2_class = self.correct_classification[s2]
                    if s2_class in max_dist_dict:
                        if self.dtwstMatrix.get_dist(s1, s2)[Constants.ZERO] > max_dist_dict[s2_class]:
                            max_dist_dict[s2_class] = self.dtwstMatrix.get_dist(s1, s2)[Constants.ZERO]
                    else:
                        max_dist_dict[s2_class] = self.dtwstMatrix.get_dist(s1, s2)[Constants.ZERO]
            classification[s1] = max_dist_dict
        return classification


    def classify_by_min_dist_connected_components(self, w=0.5):
        """
        :return: a dictionary which maps each probe id into another dictionary that is: class name -> minimum distance
                    of the probe to classes probe, changed by the connected component factor
        """
        classification = {}
        for s1 in self.dtwstMatrix.get_label_set():
            min_dist_dict = {}
            for s2 in self.dtwstMatrix.get_label_set():
                if (s1 != s2):
                    s2_class = self.correct_classification[s2]
                    d = self.dtwstMatrix.get_dist(s1, s2)
                    if d[Constants.ONE] == Constants.ZERO:
                        dist = d[Constants.ZERO]
                    else:
                        dist = d[Constants.ZERO] + (d[Constants.ZERO]*(d[Constants.ONE]/w))
                    if s2_class in min_dist_dict:
                        if self.dtwstMatrix.get_dist(s1, s2)[Constants.ZERO] < min_dist_dict[s2_class]:
                            min_dist_dict[s2_class] = dist
                    else:
                        min_dist_dict[s2_class] = dist
            classification[s1] = min_dist_dict
        return classification

    def classify_by_avg_dist(self):
        """
        :return: a dictionary which maps each probe id into another dictionary that is: class name -> average distance
                    of the probe to classes probe
        """
        classification = {}
        for s1 in self.dtwstMatrix.get_label_set():
            classes_sum_value = {}
            classes_sample_number = {}
            avg_dist_dict = {}
            for s2 in self.dtwstMatrix.get_label_set():
                if s1 != s2:
                    s2_class = self.correct_classification[s2]
                    if s2_class in classes_sum_value:
                        classes_sum_value[s2_class] += self.dtwstMatrix.get_dist(s1, s2)[Constants.ZERO]
                        classes_sample_number[s2_class] += Constants.ONE
                    else:
                        classes_sum_value[s2_class] = self.dtwstMatrix.get_dist(s1, s2)[Constants.ZERO]
                        classes_sample_number[s2_class] = Constants.ONE
            for k in classes_sum_value:
                avg_dist_dict[k] = classes_sum_value[k]/classes_sample_number[k]
            classification[s1] = avg_dist_dict
        return classification

    def classify_by_avg_dist_connected_component(self, w=0.5):
        """
        :return: a dictionary which maps each probe id into another dictionary that is: class name -> average distance
                    of the probe to classes probe, changed by the connected component factor
        """
        classification = {}
        for s1 in self.dtwstMatrix.get_label_set():
            classes_sum_value = {}
            classes_sample_number = {}
            avg_dist_dict = {}
            for s2 in self.dtwstMatrix.get_label_set():
                if s1 != s2:
                    s2_class = self.correct_classification[s2]
                    d = self.dtwstMatrix.get_dist(s1, s2)
                    if d[Constants.ONE] == Constants.ZERO:
                        dist = d[Constants.ZERO]
                    else:
                        dist = d[Constants.ZERO] + (d[Constants.ZERO]*(d[Constants.ONE]/w))
                    if s2_class in classes_sum_value:
                        classes_sum_value[s2_class] += dist
                        classes_sample_number[s2_class] += 1.0
                    else:
                        classes_sum_value[s2_class] = dist
                        classes_sample_number[s2_class] = 1.0
            for k in classes_sum_value:
                avg_dist_dict[k] = (classes_sum_value[k]/classes_sample_number[k])
            classification[s1] = avg_dist_dict
        return classification

    def evaluate_classification(self, classification):
        classes_correctness = {}
        for k in self.get_correct_classification():
            k_correctness = classification[k] == self.get_correct_classification()[k]
            if self.get_correct_classification()[k] in classes_correctness:
                if k_correctness:
                    classes_correctness[self.get_correct_classification()[k]][Constants.ZERO] += Constants.ONE
                else:
                    classes_correctness[self.get_correct_classification()[k]][Constants.ONE] += Constants.ONE
            else:
                if k_correctness:
                    classes_correctness[self.get_correct_classification()[k]] = [Constants.ONE, Constants.ZERO]
                else:
                    classes_correctness[self.get_correct_classification()[k]] = [Constants.ZERO, Constants.ONE]
        return classes_correctness

if __name__ == '__main__':
    classifier = DTWClassifier(Utils.DATASET_NAME, DTWDistMatrixManager(Utils.DATASET_NAME).get_matrix('movementPoints_filtered_by_x_y'))
    simple_min_dist_classification = classifier.classify_by_min_dist()
    for k in simple_min_dist_classification:
        print(type(k), k, type(simple_min_dist_classification[k]), simple_min_dist_classification[k])
    # single_class_value = classifier.filter_by_class(simple_min_dist_classification, 'u9_was-lx1a_0_BLOCK_LETTERS')
    # print(single_class_value)
    #
    # simple_avg_dist_classification = classifier.classify_by_min_dist()
    # single_class_value_avg = classifier.filter_by_class(simple_avg_dist_classification, 'u9_was-lx1a_0_BLOCK_LETTERS')
    # print(single_class_value_avg)
