import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

def load_data():
    data = pd.read_excel('../Data/cats_data_en.xlsx', sheet_name='Data')
    X = data.drop(columns=['Breed'])
    Y = data['Breed']
    encoder = OneHotEncoder(sparse_output=False)
    Y_onehot = encoder.fit_transform(Y.values.reshape(-1, 1))
    X = (X - X.min()) / (X.max() - X.min())
    return X.values, Y_onehot, encoder

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

class CrossEntropyCost:
    @staticmethod
    def fn(a, y, weights, lmbd):
        a = a.T
        num_samples = y.shape[0]
        cross_entropy_loss = -np.sum(y * np.log(a + 1e-10)) / num_samples
        l2_penalty = (0.5 * lmbd) / num_samples * np.sum([np.sum(w ** 2) for w in weights])
        return cross_entropy_loss + l2_penalty

    @staticmethod
    def delta(a, y):
        return a - y

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(num_neurons, 1) for num_neurons in self.sizes[1:]]
        self.weights = [np.random.randn(next_layer, current_layer) / np.sqrt(current_layer)
                        for current_layer, next_layer in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights[:-1]):
            a = np.maximum(0, np.dot(w, a) + b)
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        a = softmax(z)
        return a

    def backprop(self, X_batch, Y_batch):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        m = X_batch.shape[0]
        activations = [X_batch.T]
        zs = []
        activation = X_batch.T
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = np.maximum(0, z)
            activations.append(activation)

        error = CrossEntropyCost.delta(activations[-1], Y_batch.T)
        gradient_b[-1] = np.sum(error, axis=1, keepdims=True) / m
        gradient_w[-1] = np.dot(error, activations[-2].T) / m

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = (activations[-l] > 0).astype(float)
            error = np.dot(self.weights[-l + 1].T, error) * sp
            gradient_b[-l] = np.sum(error, axis=1, keepdims=True) / m
            gradient_w[-l] = np.dot(error, activations[-l - 1].T) / m

        return gradient_b, gradient_w

    def evaluate(self, X, Y):
        output = self.feedforward(X.T)
        predicted_labels = np.argmax(output, axis=0)
        true_labels = np.argmax(Y, axis=1)
        accuracy = np.mean(predicted_labels == true_labels) * 100
        return accuracy

    def train(self, X_train, Y_train, epochs, batch_size, learning_rate, lmbd=0.0005, val_X=None, val_Y=None,
              patience=10):
        best_val_accuracy = 0
        epochs_without_improvement = 0
        train_errors = []
        val_errors = []
        train_accuracies = []
        val_accuracies = []
        misclassified_points = []

        for epoch in range(epochs):
            X_train, Y_train = shuffle(X_train, Y_train)

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                Y_batch = Y_train[i:i + batch_size]

                gradient_b, gradient_w = self.backprop(X_batch, Y_batch)
                self.weights = [(1 - learning_rate * (lmbd / len(X_train))) * w - learning_rate * gw
                                for w, gw in zip(self.weights, gradient_w)]
                self.biases = [b - learning_rate * gb for b, gb in zip(self.biases, gradient_b)]

            train_cost = CrossEntropyCost.fn(self.feedforward(X_train.T), Y_train, self.weights, lmbd)
            train_errors.append(train_cost)
            train_accuracy = self.evaluate(X_train, Y_train)
            train_accuracies.append(train_accuracy)

            if val_X is not None and val_Y is not None:
                val_cost = CrossEntropyCost.fn(self.feedforward(val_X.T), val_Y, self.weights, lmbd)
                val_errors.append(val_cost)
                val_accuracy = self.evaluate(val_X, val_Y)
                val_accuracies.append(val_accuracy)

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break

        return train_accuracies, val_accuracies

def save_model(network, filename):
    with open(filename, 'wb') as f:
        pickle.dump(network, f)

# 6. Function to load the model
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 7. Function to predict breed from input data
def predict_breed(network, input_data, breed_mapping):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
    predicted_breed = network.feedforward(input_data.T)
    predicted_label = np.argmax(predicted_breed, axis=0)
    predicted_label = predicted_label.item()
    predicted_breed_name = breed_mapping.get(predicted_label, "Unknown breed")
    return predicted_breed_name

# Main function to run the training process
def main():
    file_path = '../Data/cats_data_en.xlsx'
    train_X, train_Y, encoder = load_data()
    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.2)

    net = Network([train_X.shape[1], 512, train_Y.shape[1]])

    epochs = 100
    batch_size = 128
    learning_rate = 0.1
    lmbd = 0.0001

    train_accuracies, val_accuracies = net.train(train_X, train_Y, epochs, batch_size, learning_rate, lmbd, val_X, val_Y)

    save_model(net, 'model.pkl')

    breed_mapping = {
        1: "Bengal", 2: "Birman", 3: "British Shorthair", 4: "Chartreux", 5: "European", 6: "Maine Coon",
        7: "Persian", 8: "Ragdoll", 9: "Savannah", 10: "Sphynx", 11: "Siamese", 12: "Turkish Angora", 0: "Other",
        -1: "Not specified", -2: "No breed"
    }

    input_data = pickle.load(open("input_data.pkl", "rb"))
    predicted_breed_name = predict_breed(net, input_data, breed_mapping)

    print(f"The predicted breed is: {predicted_breed_name}")
    print(f"Final training accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")

