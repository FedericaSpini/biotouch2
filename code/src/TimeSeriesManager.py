import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
logging.basicConfig(level=logging.ERROR)

import tsfresh
import pandas


import src.DataManager as dm
import src.Chronometer as Chronom
import src.Utils as Utils



class TimeSeriesManager:

    @staticmethod
    def _check_saved_pickles(dataset_name):
        for label in Utils.TIMED_POINTS_SERIES_TYPE:
            if not Utils.os.path.isfile(Utils.BUILD_FEATURE_PICKLE_PATH(dataset_name, label)):
                return False
        return True

    # @staticmethod
    # def extract_series_from_dataframe(dataframe: pandas.DataFrame, wordid_userid_mapping):
        # res = {}
        # series_split_by_word_id = []
        #
        # res(label) = series_split_by_word_id
        # return tsfresh.extract_relevant_features(dataframe, wordid_userid_mapping,
        #                                          column_id=Utils.WORD_ID, column_sort=Utils.TIME, n_jobs=4)

    def __init__(self, dataset_name, update_data=False, update_features=False, anonymous=False):
        update_features = update_features or update_data

        self.dataset_name = dataset_name
        self.data_frames = {}
        self.data_time_series = {}

        self._load_time_series(update_data, update_features)

    def get_features(self):
        return self.data_time_series

    def get_classes(self):
        """
        :return: a pandas' Series object that for each sample
        numeric id contains the correct class
        """
        return self.data_frames[Utils.WORDID_USERID]

    def get_classes_data(self):
        """
        :return: a pandas' DataFrame object that for each class stores a set of information.
        For a Biotouch dataset these are: ['age', 'deviceFingerPrint', 'deviceModel', 'heigthPixels',
       'widthPixels', 'xdpi', 'ydpi', 'gender', 'handwriting', 'id', 'name',
       'surname', 'totalWordNumber']
        """
        return self.data_frames[Utils.USERID_USERDATA]

    # def _load_features(self, update_data, update_features, anonymous=True):
    #     if anonymous:
    #         self.data_frames = dm.AnonymousDataManager(self.dataset_name, update_data).get_dataframes()
    #     else:
    #         self.data_frames = dm.DataManager(self.dataset_name, update_data).get_dataframes()
    #     if not update_features and FeaturesManager._check_saved_pickles(self.dataset_name):
    #         self._read_pickles()
    #         return
    #     else:
    #         self._extract_features_from_dataframes()
    #         self._load_features(False, False)

    def _load_time_series(self, update_data, update_series):
        self.data_frames = dm.AnonymousDataManager(self.dataset_name, update_data).get_dataframes()
        if not update_series and TimeSeriesManager._check_saved_pickles(self.dataset_name):
            self._read_pickles()
            return
        else:
            self._extract_series_from_dataframes()
            self._load_features(False, False)

    def _read_pickles(self):
        chrono = Chronom.Chrono("Reading series pickles...")
        for label in Utils.TIMED_POINTS_SERIES_TYPE:
            self.data_time_series[label] = pandas.read_pickle(Utils.BUILD_FEATURE_PICKLE_PATH(self.dataset_name, label))
        chrono.end()

    def _extract_series_from_dataframes(self):
        words_id_list = list(range(len(self.data_frames[Utils.WORDID_USERID])))
        for label in Utils.POINTS_SERIES_TYPE:
            chrono = Chronom.Chrono("Extracting time series from {}...".format(label), True)
            series_split_by_word_id = []
            for i in words_id_list:
                df = self.data_frames[label]
                series_split_by_word_id.append(df[df['word_id']==i])
            self.data_time_series[label] = series_split_by_word_id
            chrono.end()
        # print(self.data_time_series['movementPoints'][0], '    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(self.data_time_series['movementPoints'][0], '    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(len(self.data_time_series))

    def _save_features(self, to_csv=True):
        Utils.save_dataframes(self.dataset_name, self.data_time_series, Utils.FEATURE, "Saving features...",
                              to_csv, Utils.TIMED_POINTS_SERIES_TYPE, self.data_frames[Utils.WORDID_USERID])

    def _save_feature(self, dict_to_save, to_csv=True):
        Utils.save_dataframes(self.dataset_name, dict_to_save, Utils.FEATURE, "Saving features...",
                              to_csv, Utils.TIMED_POINTS_SERIES_TYPE, self.data_frames[Utils.WORDID_USERID])


if __name__ == '__main__':
    TimeSeriesManager(Utils.MINI_DATASET_NAME, update_data=True, update_features=True)
