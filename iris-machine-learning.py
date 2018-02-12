import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

import keras
from keras import models
from keras import layers
from keras.utils import to_categorical


def main():

    # Analyze the data-set.
    analyze()

    # Apply Machine Learning.
    machine_learning()

    # Apply Deep Learning.
    deep_learning()


def analyze():

    print("Analyzing data-set...")

    iris_dataset = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    iris_dataframe = pd.DataFrame(
        x_train,
        columns=iris_dataset.feature_names
    )
    pd.plotting.scatter_matrix(
        iris_dataframe, c=y_train,
        figsize=(15,15), marker="o",
        hist_kwds={"bins": 20}, s=60, alpha=0.8, cmap=mglearn.cm3)
    plt.show()

    print("")


def machine_learning():

    print("Applying k-Neighbors algorithm...")

    # Loading the data-set.
    iris_dataset = load_iris()

    # Train-test-split.
    x_train, x_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)

    # Initializing the classifier.
    knn = KNeighborsClassifier(n_neighbors=1)

    # Training the classifier.
    knn.fit(x_train, y_train)

    # Evaluating the model.
    test_accuracy = knn.score(x_test, y_test)
    print("k-Neighbors test-accuracy:", test_accuracy)
    print("")


def deep_learning():

    print("Applying Deep Learning...")

    # Loading the data-set.
    iris_dataset = load_iris()
    input_data = iris_dataset["data"]
    output_data = iris_dataset["target"]

    # Applying a to_categorical encoding.
    output_data = to_categorical(output_data)

    # Train-test-split.
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, random_state=0)

    # Normalizing the data.
    minimum = np.min(x_train)
    maximum = np.max(x_train)
    x_train = (x_train - minimum) / (maximum - minimum)
    x_test = (x_test - minimum) / (maximum - minimum)


    # Creating the model.
    model = models.Sequential()
    model.add(layers.Dense(40, input_shape=(4,)))
    model.add(layers.Dense(3, activation="softmax"))

    # Compiling the model.
    model.compile(
        loss="mse",
        optimizer="rmsprop",
        metrics=["accuracy"]
    )

    # Training the model.
    model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=0)

    # Evaluating the model.
    _, test_accuracy = model.evaluate(x_test, y_test)
    print("Deep Learning test-accuracy:", test_accuracy)
    print("")


if __name__ == "__main__":
    main()
