#!/usr/bin/env python

from confusion_matrix import ConfusionMatrix

def log_confusion_matrix(log, cm):
    for fmt in ConfusionMatrix(cm).as_text_lines():
        log.info(fmt)
