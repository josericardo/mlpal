#!/bin/env python
# coding=utf-8

import json
import os

class History:
    def __init__(self, id='history'):
        self.file = '%s.json' % id

        if os.path.isfile(self.file):
            self.entries = json.loads(open(self.file).read())
        else:
            self.entries = []

    def new(self):
        return {}

    def append(self, new_entry):
        self.entries.append(new_entry)
        self._save()

    def _save(self):
        json.dump(self.entries, open(self.file, 'w'))

    def get(self, i):
        return self.entries[i]

    def __len__(self):
        return len(self.entries)

    def erase(self):
        os.remove(self.file)
