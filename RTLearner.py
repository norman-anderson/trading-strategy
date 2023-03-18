import numpy as np
import random


class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.decision_tree = None

    def author(self):
        return "nanderson83"

    def add_evidence(self, data_x, data_y):
        self.decision_tree = self.build_tree(data_x, data_y)
        # edge case: if our tree is only a leaf, convert it to 2D array
        if isinstance(self.decision_tree[0], int) \
                or isinstance(self.decision_tree[0], float):
            self.decision_tree = np.array([self.decision_tree])
        if self.verbose:
            print(self.decision_tree)

    def build_tree(self, data_x, data_y):
        # convert data_x and data_y to astype(float) to be able to take median
        data_x = data_x.astype(float)
        data_y = data_y.astype(float)

        # base cases
        if data_x.shape[0] <= self.leaf_size:
            return np.array([-1, np.median(data_y), np.nan, np.nan])
        if np.all(data_y == data_y[0]): # if all data_y are the same
            return np.array([-1, data_y[0], np.nan, np.nan])

        feature_i = random.randrange(data_x.shape[1])
        point1, point2 = random.sample(range(data_x.shape[0]), 2)
        split_val = (data_x[point1][feature_i] + data_x[point2][feature_i]) / 2

        # edge case:
        if split_val == max(data_x[:, feature_i]):
            return np.array([-1, np.median(data_y), np.nan, np.nan])

        left_tree = self.build_tree(data_x[data_x[:, feature_i] <= split_val],
                                    data_y[data_x[:, feature_i] <= split_val])
        right_tree = self.build_tree(data_x[data_x[:, feature_i] > split_val],
                                     data_y[data_x[:, feature_i] > split_val])

        # assemble the root to have
        # [best index, split value, relative left subtree, relative right subt.]
        if left_tree.ndim == 1:
            root = np.array([feature_i, split_val, 1, 2])
        else:
            root = np.array([feature_i, split_val, 1, left_tree.shape[0] + 1])
        return np.row_stack((root, left_tree, right_tree))

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @return: the estimated values according to the saved model.
        """
        result = np.array([])
        for point in points:
            # edge case: if our tree is a single leaf
            if isinstance(self.decision_tree[0], int) \
                    or isinstance(self.decision_tree[0],float):
                result = np.append(result, self.decision_tree[1])
                continue

            node = 0
            feature = self.decision_tree[node][0]
            while feature != -1:
                split_val = self.decision_tree[node][1]
                if point[int(float(feature))] <= float(split_val):
                    # go to the left subtree
                    node += 1
                else:
                    # go to the right subtree
                    node += int(float(self.decision_tree[node][3]))
                # update feature
                feature = self.decision_tree[node][0]

            # we have reached a leaf
            value = self.decision_tree[node][1]
            result = np.append(result, value)
        return result
