import numpy as np
import RTLearner as rt
import math


class BagLearner(object):
    def __init__(self,  bags, learner, kwargs,  boost=False, verbose=False):
        super(BagLearner, self).__init__()
        self.learners = np.array([])
        self.bags = bags
        for _ in range(self.bags):
            self.learners = np.append(self.learners, learner(**kwargs))
        self.kwargs = kwargs

    def author(self):
        return "nanderson83"

    def add_evidence(self, data_x, data_y):
        def generate_train_data(data_x, data_y):
            """
            Helper function to generate train data, moved in so Insane Learner
            doesn't inherit it.
            """
            combined_data = np.ones((data_x.shape[0], data_x.shape[1] + 1))
            for i in range(data_y.shape[0]):
                combined_data[i] = np.append(data_x[i], data_y[i])

            random_no = np.random.choice(combined_data.shape[0],
                                         size=int(0.6 * combined_data.shape[0]),
                                         replace=True)
            test_rows_no = np.random.choice(random_no, size=combined_data.shape[0],
                                            replace=True)

            train_combined_data = np.array([])
            for j in test_rows_no:
                if len(train_combined_data) > 0:
                    train_combined_data = np.row_stack((train_combined_data,
                                                        combined_data[j]))
                else:
                    train_combined_data = combined_data[j]

            return train_combined_data[:, :-1], train_combined_data[:, -1]

        learner_result = np.array([])
        for learner in self.learners:
            train_x, train_y = generate_train_data(data_x, data_y)
            learner_result = np.append(learner_result,
                                       learner.add_evidence(train_x, train_y))

        # return self to be used in InsaneLearner
        return self

    def query(self,testX):
        result = np.array([])
        for i in range(self.bags):
            if len(result) > 0:
                result = np.row_stack((result, self.learners[i].query(testX)))
            else:
                result = np.array([self.learners[i].query(testX)])
        return np.mean(result, axis=0)
