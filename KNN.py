def euclidean_distance(x1, x2):
    return sum((x1_i - x2_i) ** 2 for x1_i, x2_i in zip(x1, x2)) ** 0.5

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return max(set(k_nearest_labels), key=k_nearest_labels.count)

# dataset
X_train = [
    [1, 2], [5, 3], [9, 1],
    [3, 5], [8, 7], [4, 6]
]
y_train = [0, 0, 0, 1, 1, 1]

X_test = [[5, 5], [1, 1]]

# Tpredict
knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print("Predictions:", predictions)
