# Example 4.1 Neuron.py (Modified for Logical OR)

from numpy import exp, array, random, dot

# Set up parameters ===============================================
inputs = array([[0, 0], [1, 1], [1, 0], [0, 1]])
outputs = array([[0, 1, 1, 1]]).T  # OR truth table: 0|0=0, 1|1=1, 1|0=1, 0|1=1
random.seed(1)
weights = 2 * random.random((2, 1)) - 1

# Define a Single Neuron function =================================
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output

# Train the Neuron ================================================
for iteration in range(50000):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

# Test the Neuron =================================================
print("Final weights:", weights)
print("\nTesting OR operation:")
print("0 OR 0 =", neuron(array([[0, 0]]), weights)[0][0])
print("1 OR 1 =", neuron(array([[1, 1]]), weights)[0][0])
print("1 OR 0 =", neuron(array([[1, 0]]), weights)[0][0])
print("0 OR 1 =", neuron(array([[0, 1]]), weights)[0][0])

# Round the outputs for clearer results
print("\nRounded outputs (threshold 0.5):")
print("0 OR 0 =", round(neuron(array([[0, 0]]), weights)[0][0]))
print("1 OR 1 =", round(neuron(array([[1, 1]]), weights)[0][0]))
print("1 OR 0 =", round(neuron(array([[1, 0]]), weights)[0][0]))
print("0 OR 1 =", round(neuron(array([[0, 1]]), weights)[0][0]))