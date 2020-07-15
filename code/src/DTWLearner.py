from datetime import datetime

import sklearn
import sklearn.model_selection
import pandas
import numpy

import src.Constants
import random
import src.TimeSeriesManager as fm
from src import Utils

LEARNING_FROM = Utils.TIMED_POINTS_SERIES_TYPE
MOVEMENT_WEIGHT = 0.75


class DTWWordClassifier:

    @staticmethod
    def scale_features(x_train, x_test):
        """
        Normalizes with average 0 and variancy 1 the features thank to sklearn.
        Then train on the training set and applies the transformation on the test set
        :param x_train:
        :param x_test:
        :return:
        """
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(x_train)
        return scaler.transform(x_train), scaler.transform(x_test)

    @staticmethod
    def filter_by_handwriting(dataframe, classes, user_data, handwriting):
        """
        (maybe) Estrae dal dataset contenuto in dataframe i sample di user con il dato handwriting sytle
        Forse meglio stampare il dataframe in input e in output.

        :param dataframe:
        :param classes:
        :param user_data:
        :param handwriting:
        :return:
        """
        d_temp = dataframe.join(classes).join(user_data[Utils.HANDWRITING], on=Utils.USER_ID).drop(Utils.USER_ID,
                                                                                                   axis=1)
        dataframe_filt = d_temp[d_temp[Utils.HANDWRITING] == handwriting].drop(Utils.HANDWRITING, axis=1)

        c_temp = pandas.DataFrame(classes).join(user_data[Utils.HANDWRITING], on=Utils.USER_ID)
        classes_filt = c_temp[c_temp[Utils.HANDWRITING] == handwriting].drop(Utils.HANDWRITING, axis=1).squeeze()

        return dataframe_filt, classes_filt


    def split_scale_data(self, label, handwriting, test_size, stratify=True, random_state=None):
        """
        Splits the data in training and test set, and normalizes.
        The features are in self.features.
        :param label: indicates wich feature in self.features should be selected (e.g. TIMED_POINTS_SERIES_TYPE)
        :param handwriting: e.g. 'ITALIC' or 'BLOCK_LETTERS'
        :param test_size: the percentage of data that must be served for test and not for training
        :param stratify:
        :param random_state:
        :return: The training test and the test set
        """
        x, y = DTWWordClassifier.filter_by_handwriting(self.features[label], self.classes, self.classes_data, handwriting)
        xtrain, xtest, y_train, y_test = sklearn.model_selection.train_test_split(x, y,
                                                                                  stratify=y if stratify else None,
                                                                                  random_state=random_state,
                                                                                  test_size=test_size)
        X_train, X_test = DTWWordClassifier.scale_features(xtrain, xtest)
        # X_train is the training set dataframe, y_train the series of the correct training set classes
        # X_test is the testing set dataframe, y_test the series of the correct testing set classes
        return X_train, X_test, y_train, y_test


    def __init__(self, dataset_name, handwriting, test_size=0.3125, update_data=False, check_consistency=False, weight=MOVEMENT_WEIGHT, autofit=True, anonymous=True):
        """
        This is the constructor of a WordClassifier object.
        :param dataset_name: a string, the name of the biotouch dataset, stored into the res folder, on wich the classification should be done
        :param handwriting: a string, generally "ITALIC" or "BLOCK_LETTERS"
        :param test_size: the percentage of data that must be used to test and not to train the classifier
        :param update_data: a boolean which deserves to build the FeatureManager
        :param check_consistency:
        :param weight:
        :param autofit:
        """
        self.dataset_name = dataset_name
        self.handwriting = handwriting

        #the object wich manages extracts features
        self.feature_manager = fm.TimeSeriesManager(dataset_name, update_data, anonymous=anonymous)

        # features is a dictionary which maps the type of temporary series of points
        # (for example "xy_shifted_touchDownPoints" to a pandas' dataframe table, that
        # for each sample id contains a lot of features value)
        self.features = self.feature_manager.get_features()
        self.classes = self.feature_manager.get_classes() # a pandas' Series that lists the classes of the samples
        self.classes_data = self.feature_manager.get_classes_data() # a pandas DataFrame that store the classes features (as 'age', 'gender')

        self.X_train = {x: None for x in LEARNING_FROM}
        self.X_test = {x: None for x in LEARNING_FROM}
        self.y_train = None
        self.y_test = None

        self.check_inconsistency = check_consistency

        random.seed(datetime.now())
        #setting r to a deterministic value result among reruns are less random
        r = 0  # random.randint(0, 10000)

        for label in LEARNING_FROM:  # LEARNING_FROM contains the names of the various time series to consider

            # a is the dataframe with the training set data, c the series with its correct label
            # b is the dataframe with the testing set data, d the series with its correct label

            a, b, c, d = self.split_scale_data(label, handwriting, test_size, random_state=r)
            self.X_train[label], self.X_test[label] = a, b      #The type of time series (e.g. label touchUpPoints) are associated in these dictionaries
                                                                # to the correct array of training and testing values
            chiave_casuale = random.choice(list(self.X_train.keys()))

            # y_train is the series of strings representing the correct classes for X_train data
            # y_test is the series of strings representing the correct classes for X_test data
            if self.y_train is not None or self.y_test is not None:
                assert (self.y_train == c).all()
                assert (self.y_test == d).all()
            else:
                self.y_train = c
                self.y_test = d

        # self.svms = {}
        # self.mov_weight = weight

        from typing import Mapping, Callable, List

        # Quado uso List è perchè non so che tipo
        # Con SVC intendo "predittore", quindi anche majority vote di più svm
        # self.predict_fun = ???
        # self.predict_proba_fun = ???
        # self.predict_functions: Mapping[str,
        #                                 Callable[
        #                                     [
        #                                         Mapping[str, SVC],
        #                                         Mapping[str, List]],
        #                                     List]] = {
        #     MOVEMENT: lambda svms, xtest: svms[MOVEMENT].predict(xtest[MOVEMENT]),
        #     UP: lambda svms, xtest: svms[UP].predict(xtest[UP]),
        #     DOWN: lambda svms, xtest: svms[DOWN].predict(xtest[DOWN]),
        #     MAJORITY: lambda svms, xtest: WordClassifier.majority_vote(
        #         (svms[x].predict(xtest[x]) for x in [MOVEMENT, UP, DOWN])),
        #     AVERAGE: lambda svms, xtest: self.max_proba_class(self.predict_proba_functions[AVERAGE](svms, xtest)),
        #     WEIGHTED_AVERAGE: lambda svms, xtest: self.max_proba_class(
        #         self.predict_proba_functions[WEIGHTED_AVERAGE](svms, xtest)),
        #
        #     XY_MOVEMENT: lambda svms, xtest: svms[XY_MOVEMENT].predict(xtest[XY_MOVEMENT]),
        #     XY_UP: lambda svms, xtest: svms[XY_UP].predict(xtest[XY_UP]),
        #     XY_DOWN: lambda svms, xtest: svms[XY_DOWN].predict(xtest[XY_DOWN]),
        #     XY_MAJORITY: lambda svms, xtest: WordClassifier.majority_vote(
        #         (svms[x].predict(xtest[x]) for x in [XY_MOVEMENT, XY_UP, XY_DOWN])),
        #     XY_AVERAGE: lambda svms, xtest: self.max_proba_class(self.predict_proba_functions[XY_AVERAGE](svms, xtest)),
        #     XY_WEIGHTED_AVERAGE: lambda svms, xtest: self.max_proba_class(
        #         self.predict_proba_functions[XY_WEIGHTED_AVERAGE](svms, xtest)),
        #
        #     ALL_MAJORITY: lambda svms, xtest: WordClassifier.majority_vote(
        #         (svms[x].predict(xtest[x]) for x in [MOVEMENT, XY_MOVEMENT, UP, XY_UP, DOWN, XY_DOWN])),
        #     ALL_AVERAGE: lambda svms, xtest: self.max_proba_class(
        #         self.predict_proba_functions[ALL_AVERAGE](svms, xtest)),
        #     ALL_WEIGHTED_AVERAGE: lambda svms, xtest: self.max_proba_class(
        #         self.predict_proba_functions[ALL_WEIGHTED_AVERAGE](svms, xtest)),
        # }

        # self.predict_proba_functions = {
        #     MOVEMENT: lambda svms, xtest: svms[MOVEMENT].predict_proba(xtest[MOVEMENT]),
        #     UP: lambda svms, xtest: svms[UP].predict_proba(xtest[UP]),
        #     DOWN: lambda svms, xtest: svms[DOWN].predict_proba(xtest[DOWN]),
        #     MAJORITY: lambda svms, xtest: self.majority_vote_proba(
        #         [svms[x].predict_proba(xtest[x]) for x in [MOVEMENT, UP, DOWN]],
        #         self.predict_functions[MAJORITY](svms, xtest)),
        #     AVERAGE: lambda svms, xtest: WordClassifier.average_proba(
        #         [svms[x].predict_proba(xtest[x]) for x in [MOVEMENT, UP, DOWN]]),
        #     WEIGHTED_AVERAGE: lambda svms, xtest: self.weighted_average_proba(
        #         [svms[x].predict_proba(xtest[x]) for x in [MOVEMENT, UP, DOWN]]),
        #
        #     XY_MOVEMENT: lambda svms, xtest: svms[XY_MOVEMENT].predict_proba(xtest[XY_MOVEMENT]),
        #     XY_UP: lambda svms, xtest: svms[XY_UP].predict_proba(xtest[XY_UP]),
        #     XY_DOWN: lambda svms, xtest: svms[XY_DOWN].predict_proba(xtest[XY_DOWN]),
        #     XY_MAJORITY: lambda svms, xtest: self.majority_vote_proba(
        #         (svms[x].predict_proba(xtest[x]) for x in [XY_MOVEMENT, XY_UP, XY_DOWN]),
        #         self.predict_functions[XY_MAJORITY](svms, xtest)),
        #     XY_AVERAGE: lambda svms, xtest: WordClassifier.average_proba(
        #         (svms[x].predict_proba(xtest[x]) for x in [XY_MOVEMENT, XY_UP, XY_DOWN])),
        #     XY_WEIGHTED_AVERAGE: lambda svms, xtest: self.weighted_average_proba(
        #         [svms[x].predict_proba(xtest[x]) for x in [XY_MOVEMENT, XY_UP, XY_DOWN]]),
        #
        #     ALL_MAJORITY: lambda svms, xtest: self.majority_vote_proba(
        #         (svms[x].predict_proba(xtest[x]) for x in [MOVEMENT, XY_MOVEMENT, UP, XY_UP, DOWN, XY_DOWN]),
        #         self.predict_functions[ALL_MAJORITY](svms, xtest)),
        #     ALL_AVERAGE: lambda svms, xtest: WordClassifier.average_proba(
        #         (svms[x].predict_proba(xtest[x]) for x in [MOVEMENT, XY_MOVEMENT, UP, XY_UP, DOWN, XY_DOWN])),
        #     ALL_WEIGHTED_AVERAGE: lambda svms, xtest: self.weighted_average_proba(
        #         [svms[x].predict_proba(xtest[x]) for x in [MOVEMENT, XY_MOVEMENT, UP, XY_UP, DOWN, XY_DOWN]])
        # }

        # if autofit:
        #     self.fit()

    def getDTWDistance(self, s1, s2):
        n = len(s1)
        m = len(s2)
        matrix = numpy.empty((n+1, m+1))
        matrix[:] = numpy.nan
        matrix[0][0] = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                if (i == 0) and (j > 0):
                    matrix[i][j] = abs(s1[i]-s2[j]) + matrix[i][j-1]
                elif (i > 0) and (j == 0):
                    matrix[i][j] = abs(s1[i] - s2[j]) + matrix[i-1][j]
                else:
                    matrix[i][j] = abs(s1[i] - s2[j]) + min(matrix[i - 1][j - 1],
                                                            min(matrix[i - 1][j],
                                                                matrix[i][j - 1]))
        return matrix[n+1][m+1]


if __name__ == '__main__':
    a = DTWWordClassifier(Utils.DATASET_NAME_3, Utils.ITALIC)
    print(a.X_test.keys(),'\n', a.X_train.keys())
    # print('X_TEST\n', type(a.X_test), len(a.X_test), '\n', a.X_test)
    print('X_TEST\n', type(a.X_test['movementPoints'][6]), len(a.X_test['movementPoints'][6]), '\n', a.X_test['movementPoints'][6])
    print('_____________________________________________________________________________________________________________')
    print('X_TRAIN\n', type(a.X_train['movementPoints'][0]),len(a.X_train['movementPoints'][0]), '\n', a.X_train['movementPoints'][0])
    print('_____________________________________________________________________________________________________________')
    print('Y_TRAIN\n', type(a.y_train), len(a.y_train),'\n',  a.y_train)
    print('_____________________________________________________________________________________________________________')
    print('Y_TEST\n', type(a.y_test),len(a.y_test), '\n', a.y_test)
    # a.fit()
    # print(a)