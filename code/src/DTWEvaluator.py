import operator

import numpy
import matplotlib.pyplot as plt


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

    def plot_cms(self, classification):
        n_classes = self.classifier.get_classes_number()
        probe_to_classified_classes = self.make_class_order_from_classification(classification)
        # print('\n\n', probe_to_classified_classes, '\n\n')
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
        label = 'CMS : '+ str(self.integrate(correct_classified_samples, 0.03125))
        plt.plot(numbers, correct_classified_samples, label=label, marker=".")
        x_label_indixes = []
        freq = 2
        for i, n in enumerate(numbers):
            if i % freq == 0:
                x_label_indixes.append(n)
        plt.xticks(x_label_indixes, numpy.arange(1, n_classes+1, freq))
        plt.yticks(numbers)
        plt.legend()
        plt.show()
        return numbers, correct_classified_samples

    def integrate(self, y_vals, h):
        i = 1
        total = y_vals[0] + y_vals[-1]
        for y in y_vals[1:-1]:
            if i % 2 == 0:
                total += 2 * y
            else:
                total += 4 * y
            i += 1
        return total * (h / 3.0)


    def plot_far_frr_1_vs_all(self, classification, threshold_min, threshold_max):
        # n_classes = self.classifier.get_classes_number()
        threshold_values = numpy.arange(threshold_min, threshold_max, 100).tolist()
        print(threshold_values)
        far_list = []
        frr_list = []
        eer = numpy.inf
        eer_threshold_index = numpy.inf
        eer_threshold_value = numpy.inf
        far_diff_from_frr = numpy.inf
        for idx, threshold in enumerate(threshold_values):
            # print(classification.keys())
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
                # eer_threshold_index = idx
                eer_threshold_value = threshold
            far_list += [false_accepted / (genuine_refused + false_accepted)]
            frr_list += [false_refused / (genuine_accepted + false_refused)]
        print(len(far_list), len(frr_list), len(threshold_values))
        print(far_list, '\n', frr_list)
        plt.plot( numpy.arange(threshold_min, threshold_max, 100), far_list, label='FAR')
        plt.plot(numpy.arange(threshold_min, threshold_max, 100), frr_list, label='FRR')
        plt.scatter(eer_threshold_value, eer, color=None, edgecolor=None)
        plt.annotate("eer: "+ str(eer)[:6], (eer_threshold_value+1000, eer))
        plt.legend()
        plt.show()


if __name__ == '__main__':
    classifier = DTWClassifier(Utils.DATASET_NAME, DTWDistMatrixManager(Utils.DATASET_NAME).get_matrix('movementPoints_filtered_by_x_y'))
    evaluator = DTWEvaluator(classifier)

    evaluator.plot_far_frr_1_vs_all(classifier.classify_by_min_dist(), 3000, 25000)
    evaluator.plot_far_frr_1_vs_all(classifier.classify_by_avg_dist(), 3000, 25000)
    evaluator.plot_far_frr_1_vs_all(classifier.classify_by_max_dist(), 3000, 25000)
    evaluator.plot_far_frr_1_vs_all(classifier.classify_by_avg_dist_connected_component(), 3000, 25000)

    # evaluator.plot_cms(classifier.classify_by_min_dist())
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=1))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=0.75))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=0.5))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=0.25))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=0.1))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=1.5))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=2))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=4))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=7.5))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=10))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=15))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=20))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=30))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=50))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=75))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=100))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=150))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=200))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=300))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=500))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=1000))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=1500))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=2000))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=3000))



    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components())

    # evaluator.plot_cms(classifier.classify_by_avg_dist())
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=1))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=0.75))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=0.5))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=0.25))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=0.1))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=0.2))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=0.1))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=1.5))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=2))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=4))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=7.5))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=10))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=15))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=20))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=30))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=50))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=75))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=100))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=150))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=200))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=300))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=500))
    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component(w=1000))


    # evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component())





