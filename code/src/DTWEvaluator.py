import operator

import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src import Utils
from src.DTWClassifier import DTWClassifier
from src.DTWDistMatrixManager import DTWDistMatrixManager


class DTWEvaluator:

    def __init__(self, classifier):
        self.classifier = classifier

    def set_classifier(self, classifier):
        self.classifier = classifier

    def make_class_order_from_classification(self, classification):
        probe_to_classified_classes = {}
        for probe in classification:
            sorted_classification = sorted(classification[probe].items(), key=operator.itemgetter(1))
            probe_to_classified_classes[probe] = sorted_classification
        return probe_to_classified_classes

    def plot_cmc(self, classification, title ='CMC'):
        n_classes = self.classifier.get_classes_number()
        probe_to_classified_classes = self.make_class_order_from_classification(classification)
        numbers = numpy.arange(1, (n_classes*1.0)+1.0)
        for i, n in enumerate(numbers):
            numbers[i] = numbers[i]/n_classes
        correct_classified_samples = numpy.zeros(n_classes)
        for probe in probe_to_classified_classes:
            correct_probe_class = self.classifier.get_correct_classification()[probe]
            correct_class_position = 0
            for c in probe_to_classified_classes[probe]:
                correct_class_position += 1
                if c[0] == correct_probe_class:
                    correct_classified_samples[correct_class_position-1] +=1
                    break
        total = 0
        for i, n in enumerate(correct_classified_samples):
            total += n
            correct_classified_samples[i] = total/1024
        label = 'AUC : '+ str(round(self.integrate(correct_classified_samples),4)) + '\nRR : ' + str(round(correct_classified_samples[0],4))
        plt.plot(numbers, correct_classified_samples, label=label, marker=".", color='brown')
        plt.fill_between(numbers, correct_classified_samples, color='#F2BEBE')
        x_label_indixes = []
        x_ticks = []
        freq = 1
        for i, n in enumerate(numbers):
            if i % freq == 0:
                x_label_indixes.append(n)
                x_ticks.append(round(n,2))
        x_ticks.append(1)
        print(x_label_indixes)
        plt.xticks(x_label_indixes, numpy.arange(1, n_classes+1, freq))
        plt.xticks(fontsize=8, rotation=60)
        plt.yticks(fontsize=8, rotation=0)
        plt.yticks(numpy.arange(0,1.1,0.1))
        plt.legend(loc=4)
        plt.title(title)
        plt.show()
        return numbers, correct_classified_samples

    def integrate(self, y_vals, h=1):
        i = 1
        total = y_vals[0] + y_vals[-1]
        for y in y_vals[1:-1]:
            if i % 2 == 0:
                total += 2 * y
            else:
                total += 4 * y
            i += 1
        return total * (h / 3.0)

    def plot_far_frr_1_vs_all(self, classification, threshold_min, threshold_max, title = 'FAR vs FRR'):
        threshold_values = numpy.arange(threshold_min, threshold_max, 100).tolist()
        print(threshold_values)
        far_list = []
        frr_list = []
        eer = numpy.inf
        eer_threshold_value = numpy.inf
        far_diff_from_frr = numpy.inf
        for idx, threshold in enumerate(threshold_values):
            genuine_accepted = 0
            genuine_refused = 0
            false_accepted = 0
            false_refused = 0
            for probe in classification.keys():
                correct_probe_class = self.classifier.get_correct_classification()[probe]
                classification_probe = classification[probe]
                for class_name in classification_probe:
                    if class_name == correct_probe_class:
                        if classification_probe[class_name] <= threshold:
                            genuine_accepted += 1
                        else:
                            false_refused += 1
                    else:
                        if classification_probe[class_name] <= threshold:
                            false_accepted +=1
                        else:
                            genuine_refused +=1
            if abs((false_accepted / (genuine_refused + false_accepted))-(false_refused / (genuine_accepted + false_refused))) < far_diff_from_frr:
                far_diff_from_frr = abs((false_accepted / (genuine_refused + false_accepted))-(false_refused / (genuine_accepted + false_refused)))
                eer = ((false_accepted / (genuine_refused + false_accepted))+(false_refused / (genuine_accepted + false_refused)))/2
                eer_threshold_value = threshold
            far_list += [false_accepted / (genuine_refused + false_accepted)]
            frr_list += [false_refused / (genuine_accepted + false_refused)]
        print(len(far_list), len(frr_list), len(threshold_values))
        print(far_list, '\n', frr_list)
        plt.plot( numpy.arange(threshold_min, threshold_max, 100), far_list, label='FAR', color='darkcyan')
        plt.plot(numpy.arange(threshold_min, threshold_max, 100), frr_list, label='FRR', color='brown')
        plt.scatter(eer_threshold_value, eer, color='red', edgecolor=None)
        plt.annotate("EER: "+ str(eer)[:6], (eer_threshold_value+1000, eer))
        plt.legend()
        plt.title(title)
        plt.xlabel('Threshold value')
        plt.show()

    def plot_roc(self, classification, threshold_min, threshold_max, title = 'ROC'):
        threshold_values = numpy.arange(threshold_min, threshold_max, 1000).tolist()
        far_list = []
        tar_list = []
        for idx, threshold in enumerate(threshold_values):
            genuine_accepted = 0
            genuine_refused = 0
            false_accepted = 0
            false_refused = 0
            for probe in classification.keys():
                correct_probe_class = self.classifier.get_correct_classification()[probe]
                classification_probe = classification[probe]
                for class_name in classification_probe:
                    if class_name == correct_probe_class:
                        if classification_probe[class_name] <= threshold:
                            genuine_accepted += 1
                        else:
                            false_refused += 1
                    else:
                        if classification_probe[class_name] <= threshold:
                            false_accepted +=1
                        else:
                            genuine_refused +=1
            far_list += [false_accepted / (genuine_refused + false_accepted)]
            tar_list += [1-(false_refused / (genuine_accepted + false_refused))]
        plt.plot(far_list, tar_list, marker=",", color='darkcyan', linewidth=1)
        plt.fill_between(far_list, tar_list, color='azure')
        plt.xticks(numpy.arange(0, 1.1, 0.1))
        plt.yticks(numpy.arange(0, 1.1, 0.1))
        plt.title(title)
        plt.ylabel('1-FRR')
        plt.xlabel('FAR')
        plt.show()


if __name__ == '__main__':
    # classifier = DTWClassifier(Utils.DATASET_NAME, DTWDistMatrixManager(Utils.DATASET_NAME).get_matrix('movementPoints_filtered_by_x_y'))
    time_classifier = DTWClassifier(Utils.DATASET_NAME, DTWDistMatrixManager(Utils.DATASET_NAME, res_path=Utils.RES_FOLDER_PATH+Utils.FINAL_DTW_DISTANCES_TIME).get_matrix('movementPoints_filtered_by_time'))
    # evaluator = DTWEvaluator(classifier)
    time_evaluator = DTWEvaluator(time_classifier)

    # evaluator.plot_far_frr_1_vs_all(classifier.classify_by_min_dist(), 0, 20000, title='(x, y) time-series: minimum distance from identities FAR vs FRR')
    # evaluator.plot_far_frr_1_vs_all(classifier.classify_by_avg_dist(), 0, 20000, title='(x, y) time-series: medium distance from gallery samples FAR vs FRR')
    # evaluator.plot_cmc(classifier.classify_by_min_dist(), title='(x, y) time-series: minimum distance CMC')
    # evaluator.plot_cmc(classifier.classify_by_avg_dist(), title='(x, y) time-series: average distance CMC')
    # evaluator.plot_roc(classifier.classify_by_min_dist(), 0, 200000, title='(x, y) time-series: minimum distance from identities ROC')
    # evaluator.plot_roc(classifier.classify_by_avg_dist(), 0, 200000, title='(x, y) time-series: medium distance from identities ROC')

    # time_evaluator.plot_far_frr_1_vs_all(time_classifier.classify_by_min_dist(), 0, 20000, title='(time) time-serie: minimum distance from identities FAR vs FRR')
    # time_evaluator.plot_far_frr_1_vs_all(time_classifier.classify_by_avg_dist(), 0, 20000, title='(time) time-serie: medium distance from gallery samples FAR vs FRR')
    # time_evaluator.plot_cmc(time_classifier.classify_by_min_dist(), title='(time) time-serie: minimum distance CMC')
    # time_evaluator.plot_cmc(time_classifier.classify_by_avg_dist(), title='(time) time-serie: average distance CMC')
    # time_evaluator.plot_roc(time_classifier.classify_by_min_dist(), 0, 200000, title='(time) time-serie: minimum distance from identities ROC')
    # time_evaluator.plot_roc(time_classifier.classify_by_avg_dist(), 0, 200000, title='(time) time-serie: average distance from identities ROC')

    # evaluator.plot_far_frr_1_vs_all(time_classifier.classify_by_min_dist(), 0, 20000)

    # evaluator.plot_far_frr_1_vs_all(classifier.classify_by_avg_dist(), 3000, 25000)
    # evaluator.plot_far_frr_1_vs_all(classifier.classify_by_max_dist(), 3000, 25000)
    # evaluator.plot_far_frr_1_vs_all(classifier.classify_by_avg_dist_connected_component(), 3000, 25000)

    # evaluator.plot_far_frr_1_vs_all(classifier.classify_by_min_dist_connected_components(w=500), 3000, 25000, title='m500')







