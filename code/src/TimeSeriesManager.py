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

    def __init__(self, dataset_name, update_data=False, update_features=False):
        update_features = update_features or update_data

        self.dataset_name = dataset_name
        self.data_frames = {}
        self.data_series = {}

        self._load_time_series(update_data, update_features)

    def get_series(self):
        return self.data_series

    def get_classes(self):
        return self.data_series[Utils.WORDID_USERID]

    def _load_time_series(self, update_data, update_series):
        self.data_frames = dm.AnonymousDataManager(self.dataset_name, update_data).get_dataframes()
        self._extract_series_from_dataframes()

    def _read_pickles(self):
        chrono = Chronom.Chrono("Reading series pickles...")
        for label in Utils.TIMED_POINTS_SERIES_TYPE:
            self.data_series[label] = pandas.read_pickle(Utils.BUILD_FEATURE_PICKLE_PATH(self.dataset_name, label))
        chrono.end()

    def _extract_series_from_dataframes(self):
        self.data_series = {}
        words_id_list = list(range(len(self.data_frames[Utils.WORDID_USERID])))
        for label in Utils.POINTS_SERIES_TYPE:
            chrono = Chronom.Chrono("Extracting time series from {}...".format(label), True)
            series_split_by_word_id = []
            for i in words_id_list:
                df = self.data_frames[label]
                series_split_by_word_id.append(df[df['word_id']==i])
            self.data_series[label] = series_split_by_word_id
            chrono.end()
        self.data_series[Utils.WORDID_USERID] = self.data_frames[Utils.WORDID_USERID].tolist()


if __name__ == '__main__':
    TimeSeriesManager(Utils.DATASET_NAME, update_data=True, update_features=True)
