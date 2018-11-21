import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import pickle
import json
import requests
from flask import jsonify


class PreProcess:
    labels = []
    words = []

    train_labels = []
    train_words = []

    test_labels = []
    test_words = []

    test_data_size = 0.2

    def __init__(self, test_num, filepath):
        self.test_tot = test_num
        self.DATA_FILEPATH = filepath

    def read_from_file(self):
        tot = self.test_tot
        with open(self.DATA_FILEPATH) as f:
            reader = csv.reader(f)
            for row in reader:
                if tot <= 0:
                    break
                tot = tot - 1
                self.labels.append(row[0])
                self.words.append(row[1])
                # print(reader.line_num,row[0])

    def build_train_test(self):
        self.train_labels, self.test_labels, self.train_words, self.test_words = train_test_split(
            self.labels, self.words, test_size=self.test_data_size, random_state=42)

    def extract_features(self, savefile):
        # encoder = CountVectorizer(min_df=0.1)
        encoder = TfidfVectorizer(min_df=0.1)
        self.train_features = encoder.fit_transform(self.train_words).toarray()
        self.test_features = encoder.transform(self.test_words).toarray()
        if savefile:
            with open("models/encoder.pk", "wb") as file:
                pickle.dump(encoder, file)

    def run(self):
        print("start process dataset")
        self.read_from_file()
        self.build_train_test()
        self.extract_features(False)


if __name__ == '__main__':
    p = PreProcess(200, "dataset/shuffled-full-set-hashed.csv")
    p.run()

