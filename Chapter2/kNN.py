#!/usr/bin/env python
import math
import operator
class kNN(object):
    # Initialization
    def __init__(self, x, y, k, weighted=False):
        assert (k <= len(x)
                ), "k cannot be greater than training_set length"
        self.__x = x
        self.__y = y
        self.__k = k
        self.__weighted = weighted
    # Compute Euclidean distance
    @staticmethod
    def __euclidean_distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    # Compute the PDF
    @staticmethod
    def gaussian(dist, sigma=1):
        return 1./(math.sqrt(2.*math.pi)*sigma)*math.exp(-dist**2/(2*sigma**2))
    # Perform predictions
    def predict(self, test_set):
        predictions = []
        for i, j in test_set.values:
            distances = []
            for idx, (l, m) in enumerate(self.__x.values):
                dist = self.__euclidean_distance(i, j, l, m)
                distances.append((self.__y[idx], dist))
            distances.sort(key=operator.itemgetter(1))
            v = 0
            total_weight = 0
            for i in range(self.__k):
                weight = self.gaussian(distances[i][1])
                if self.__weighted:
                    v += distances[i][0]*weight
                else:
                    v += distances[i][0]
                total_weight += weight
            if self.__weighted:
                predictions.append(v/total_weight)
            else:
                predictions.append(v/self.__k)
        return predictions
