import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time


def load_data():
    data = pd.read_excel('../data_augmentation/cats_data_aug.xlsx')
    X = data.drop(columns=['Race'])
    Y = data['Race']

    encoder = OneHotEncoder(sparse_output=False)
    Y_onehot = encoder.fit_transform(Y.values.reshape(-1, 1))

    X = (X - X.min()) / (X.max() - X.min())
    return X.values, Y_onehot


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

                print(f"Epoch {epoch}: "
                      f"Training cost = {train_cost:.2f}, Training accuracy = {train_accuracy:.2f}%, "
                      f"Validation cost = {val_cost:.2f}, Validation accuracy = {val_accuracy:.2f}%")

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping triggered after {epoch} epochs.")
                        break

                predicted_labels = np.argmax(self.feedforward(val_X.T), axis=0)
                true_labels = np.argmax(val_Y, axis=1)
                misclassified_indices = np.where(predicted_labels != true_labels)[0]
                misclassified_points = val_X[misclassified_indices]
            else:
                print(f"Epoch {epoch}: "
                      f"Training cost = {train_cost:.2f}, Training accuracy = {train_accuracy:.2f}%")

        self.plot_convergence(train_errors, val_errors, train_accuracies, val_accuracies)

        if len(misclassified_points) > 0:
            self.plot_misclassified(misclassified_points)

    @staticmethod
    def plot_convergence(train_errors, val_errors, train_accuracies, val_accuracies):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Cost", color="tab:purple")
        ax1.plot(train_errors, label="Training Cost", color="green", linestyle="-")
        if val_errors:
            ax1.plot(val_errors, label="Validation Cost", color="magenta", linestyle="--")
        ax1.tick_params(axis="y", labelcolor="tab:purple")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy (%)", color="tab:orange")
        ax2.plot(train_accuracies, label="Training Accuracy", color="blue", linestyle=":")
        if val_accuracies:
            ax2.plot(val_accuracies, label="Validation Accuracy", color="red", linestyle="-.")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.legend(loc="upper right")

        plt.title("Convergence of Training and Validation")
        plt.grid(True)
        plt.show()


    def plot_misclassified(self, misclassified_points):
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(misclassified_points)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color="red", label="Misclassified")
        plt.title("Misclassified Points (PCA Reduced Dimensions)")
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")
        plt.legend()
        plt.grid(True)
        plt.show()


def train_validation_split(X, Y, validation_split=0.2):
    return train_test_split(X, Y, test_size=validation_split)

start = time.time()

train_X, train_Y = load_data()

train_X, val_X, train_Y, val_Y = train_validation_split(train_X, train_Y)

net = Network([train_X.shape[1], 512, train_Y.shape[1]])

epochs = 100
batch_size = 128
learning_rate = 0.3
lmbd = 0.0002

net.train(train_X, train_Y, epochs, batch_size, learning_rate, lmbd, val_X, val_Y)

end = time.time()
print(f"Time: {(end - start) / 60:.2f} minutes")
