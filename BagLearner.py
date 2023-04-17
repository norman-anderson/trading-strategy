import numpy as np
import RTLearner as rt
import math


class BagLearner(object):
    def __init__(self,  bags, learner, kwargs,  boost=False, verbose=False):
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return "nanderson83"

    def add_evidence(self, data_x, data_y):
        rows = data_x.shape[0]
        for learner in self.learners:
            # select random indices
            i = np.random.choice(rows, size=rows)
            # select random entries
            learner.add_evidence(data_x[i], data_y[i])

    def query(self, points):
        results = []
        for learner in self.learners:
            result = learner.query(points)
            results.append(result)
        # calculate the average from each learner
        results = np.mean(np.array(results), axis=0)
        return results
