#!/usr/bin/env python

class ConfusionMatrix:
    def __init__(self, cm):
        self.cm = cm

    def predicted_as(self, claz):
        return (self.cm[0][claz] + self.cm[1][claz])

    def _abs_and_per_of_false(self, i):
        total = self.predicted_as(i)

        if total == 0:
            return (0, 0)

        false_pred = self.cm[1-i][abs(0-i)]
        perc = float(false_pred) / total
        return (false_pred, (100*perc))

    def as_text_lines(self):
        false_pos, per_fp = self._abs_and_per_of_false(1)
        false_neg, per_fn = self._abs_and_per_of_false(0)

        result = [
                "False positives: {} ({:.2f}% of all predicted as 1)".format(false_pos, per_fp),
                "False negatives: {} ({:.2f}% of all predicted as 0)".format(false_neg, per_fn)
        ]

        print result
        return result

    def as_str(self):
        return "\n".join(self.as_text_lines())

    def print_confusion_matrix(sefl):
        print self.as_str()
