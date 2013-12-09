#!/usr/bin/env python

class ConfusionMatrix:
    def __init__(self, cm):
        self.cm = cm

    def predicted_as(self, claz):
        return (self.cm[0][claz] + self.cm[1][claz])

    def as_text_lines(self):
        false_positives = self.cm[0][1]
        false_negatives = self.cm[1][0]
        per_fp = 100*(float(false_positives) / self.predicted_as(1))
        per_fn = 100*(float(false_negatives) / self.predicted_as(0))

        result = [
                "False positives: {} ({:.2f}% of all predicted as 1)".format(false_positives, per_fp),
                "False negatives: {} ({:.2f}% of all predicted as 0)".format(false_negatives, per_fn)
        ]

        return result

    def as_str(self):
        return "\n".join(self.as_text_lines())

    def print_confusion_matrix(sefl):
        print self.as_str()
