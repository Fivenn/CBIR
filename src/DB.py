# -*- coding: utf-8 -*-

from __future__ import print_function

import os

import pandas as pd


class Database(object):

    def __init__(self, DB_dir, DB_csv):
        self.DB_dir = DB_dir
        self.DB_csv = DB_csv
        self._gen_csv()
        self.data = pd.read_csv(DB_csv)
        self.classes = set(self.data["cls"])

    def _gen_csv(self):
        if os.path.exists(self.DB_csv):
            return
        with open(self.DB_csv, 'w', encoding='UTF-8') as f:
            f.write("img,cls")
            for root, _, files in os.walk(self.DB_dir, topdown=False):
                cls = root.split('/')[-1]
                for name in files:
                    if not name.endswith('.jpg'):
                        continue
                    img = os.path.join(root, name)
                    f.write("\n{},{}".format(img, cls))

    def __len__(self):
        return len(self.data)

    def get_class(self):
        return self.classes

    def get_data(self):
        return self.data


if __name__ == "__main__":
    dbTrain = Database(DB_dir="CorelDBDataSet/train", DB_csv="CorelDBDataSetTrain.csv")
    dataTrain = dbTrain.get_data()
    classesTrain = dbTrain.get_class()
    print("DB length:", len(dbTrain))
    print(classesTrain)

    dbVal = Database(DB_dir="CorelDBDataSet/val", DB_csv="CorelDBDataSetVal.csv")
    dataVal = dbVal.get_data()
    classesVal = dbVal.get_class()
    print("DB length:", len(dbVal))
    print(classesVal)
