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
        numbers = numpy.arange((n_classes*1.0)+1.0)
        for i, n in enumerate(numbers):
            numbers[i] = numbers[i]/n_classes
        correct_classified_samples = numpy.zeros(n_classes+1)
        for probe in probe_to_classified_classes:
            correct_probe_class = self.classifier.get_correct_classification()[probe]
            correct_class_position = 0
            for c in probe_to_classified_classes[probe]:
                correct_class_position += 1
                if c[0] == correct_probe_class:
                    correct_classified_samples[correct_class_position] +=1
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
        plt.xticks(x_label_indixes, numpy.arange(0, n_classes+1, freq))
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



if __name__ == '__main__':
    classifier = DTWClassifier(Utils.DATASET_NAME, DTWDistMatrixManager(Utils.DATASET_NAME).get_matrix('movementPoints_filtered_by_x_y'))
    evaluator = DTWEvaluator(classifier)
    # evaluator.plot_cms(classifier.classify_by_min_dist())
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=0.75))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=0.5))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=0.25))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=0.1))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=4))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=7.5))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=10))
    # evaluator.plot_cms(classifier.classify_by_min_dist_connected_components(w=15))

    # evaluator.plot_cms(classifier.classify_by_avg_dist())
    evaluator.plot_cms(classifier.classify_by_avg_dist_connected_component())




