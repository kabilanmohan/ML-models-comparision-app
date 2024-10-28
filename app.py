import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Function to load dataset based on user's choice
def load_data(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    elif dataset_name == "Digits":
        data = datasets.load_digits()
    return data

# Function to train and evaluate RandomForest or SVM and plot for SVM
def train_and_plot_svm(data, kernel='linear'):
    X = data.data[:, :2]  # Using only the first two features for visualization
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Plot decision boundary for SVM
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title(f"SVM ({kernel} kernel) Decision Boundary")

    return accuracy, plt

# Function to train and evaluate RandomForest
def train_random_forest(data):
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("Classifier Comparison App")

# Select Dataset
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Wine", "Digits"))

# Select Classifier
classifier_name = st.sidebar.selectbox("Select Classifier", ("SVM","Random Forest"))

# For SVM, select kernel type
if classifier_name == "SVM":
    kernel = st.sidebar.selectbox("Select Kernel", ("linear", "poly", "rbf", "sigmoid"))
else:
    kernel = None

# Load the dataset
data = load_data(dataset_name)
st.write(f"## Dataset: {dataset_name}")
st.write(f"Number of samples: {data.data.shape[0]}, Number of features: {data.data.shape[1]}")

# Train and evaluate based on the selected classifier
if classifier_name == "Random Forest":
    accuracy = train_random_forest(data)
    st.write(f"### Random Forest Accuracy: {accuracy * 100:.2f}%")
else:
    accuracy, plt_figure = train_and_plot_svm(data, kernel)
    st.write(f"### SVM ({kernel} kernel) Accuracy: {accuracy * 100:.2f}%")
    st.pyplot(plt_figure)
