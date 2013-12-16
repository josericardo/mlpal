#!/bin/env python
# coding=utf-8

import json
import os

class History:
    def __init__(self, file='history.json'):
        self.file = file

        if os.path.isfile(file):
            self.entries = json.loads(open(file).read())
        else:
            self.entries = []

    def new(self):
        return {}

    def append(self, new_entry):
        self.entries.append(new_entry)
        self._save()

    def _save(self):
        json.dump(self.entries, open(self.file, 'w'))

    def __len__(self):
        return len(self.entries)

    def erase(self):
        os.remove(self.file)
