"""Unsupervised nearest neighbors learner"""
# Source: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

# Example of making predictions
from math import sqrt

class NearestNeighbors():
    # calculate the Euclidean distance between two vectors
    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)


    # Locate the most similar neighbors
    def get_neighbors(self, train, test_row, num_neighbors):
        distances = list()
        for train_row in train:
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    # Make a classification prediction with neighbors
    def predict_classification(self, train, test_row, num_neighbors):
        neighbors = self.get_neighbors(train, test_row, num_neighbors)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction

# Test distance function
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

nn = NearestNeighbors()

# neighbors = nn.get_neighbors(dataset, dataset[0], 6)
# for neighbor in neighbors:
#     print(neighbor)

prediction = nn.predict_classification(dataset, dataset[0], 6)
print('Expected %d, Got %d.' % (dataset[0][-1], prediction))

# class NearestNeighbors(NeighborsBase, KNeighborsMixin,
#                        RadiusNeighborsMixin, UnsupervisedMixin):
# # class NearestNeighbors():
#     def __init__(self, n_neighbors=5, radius=1.0,
#                  algorithm='kd_tree', leaf_size=30, metric='minkowski',
#                  p=2, metric_params=None, n_jobs=None, **kwargs):
#         super().__init__(
#               n_neighbors=n_neighbors,
#               radius=radius,
#               algorithm=algorithm,
#               leaf_size=leaf_size, metric=metric, p=p,
#               metric_params=metric_params, n_jobs=n_jobs, **kwargs)
