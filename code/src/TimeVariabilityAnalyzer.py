from src import Utils
from src.DataManager import AnonymousDataManager
from src.TimeSeriesManager import TimeSeriesManager


class TimeVariabilityAnalyzer:

    def __init__(self, dataset_name, update_data=False):
        self.dataset_name = dataset_name
        self.time_series_manager = TimeSeriesManager(dataset_name, update_data)
        self.datamanager = AnonymousDataManager(dataset_name, update_data)
        print(type(d.data_frames['movementPoints']['time']), d.data_frames['movementPoints']['time'])



if __name__ == '__main__':
    d = AnonymousDataManager(Utils.MINI_DATASET_NAME, update_data=False)
    print(type(d.data_frames['movementPoints']['time']), d.data_frames['movementPoints']['time'])

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
