# Example 4.2 Multiple Layer Perceptron (Modified for 3 inputs)
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.neural_network import MLPClassifier

# Three inputs instead of two - all 8 possible combinations
X = [[0., 0., 0.],
     [0., 0., 1.],
     [0., 1., 0.],
     [0., 1., 1.],
     [1., 0., 0.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.]]

# Target output: OR of the three inputs
# (0,0,0)->0, and any input containing at least one 1 -> 1
y = [0, 1, 1, 1, 1, 1, 1, 1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

# Test with new three-input samples
print("Predictions on new samples:")
print(clf.predict([[1., 1., 1.], [0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]))

# Show the shapes of the weight matrices
print("\nWeight matrix shapes:", [coef.shape for coef in clf.coefs_])

# Optional: verify performance on training data
print("\nPredictions on training data:", clf.predict(X))
print("Actual targets:            ", y)