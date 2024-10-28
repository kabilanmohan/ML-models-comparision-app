import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(
    page_title="ML Classifier Comparison",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Function to load dataset
def load_data(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    elif dataset_name == "Digits":
        data = datasets.load_digits()
    return data

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return fig

# Function for Random Forest visualization
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    return fig

# Function to train and plot SVM
def train_and_plot_svm(data, kernel='linear'):
    X = data.data[:, :2]
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)  # Fixed 'edcolor' to 'edgecolors'
    plt.colorbar(scatter)
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title(f"SVM ({kernel} kernel) Decision Boundary")
    
    return accuracy, fig, y_test, y_pred

# Function to train Random Forest
def train_random_forest(data):
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, clf, y_test, y_pred

# Main content area
st.title("🤖 Machine Learning Classifier Comparison")

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    # Dataset selection
    dataset_name = st.selectbox(
        "Select Dataset",
        ("Iris", "Wine", "Digits"),
        help="Choose the dataset you want to analyze"
    )

    # Classifier selection
    classifier_name = st.selectbox(
        "Select Classifier",
        ("SVM", "Random Forest"),
        help="Choose the machine learning algorithm"
    )

# Load data
data = load_data(dataset_name)

# Display dataset info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Number of Samples", data.data.shape[0])
with col2:
    st.metric("Number of Features", data.data.shape[1])
with col3:
    st.metric("Number of Classes", len(np.unique(data.target)))

# Training and visualization
if classifier_name == "SVM":
    with st.sidebar:
        kernel = st.selectbox("Select Kernel", ("linear", "poly", "rbf", "sigmoid"))
    
    accuracy, fig, y_test, y_pred = train_and_plot_svm(data, kernel)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Decision Boundary")
        st.pyplot(fig)
    with col2:
        st.markdown(f"### Confusion Matrix")
        cm_fig = plot_confusion_matrix(y_test, y_pred, classes=data.target_names)
        st.pyplot(cm_fig)

else:  # Random Forest
    accuracy, rf_model, y_test, y_pred = train_random_forest(data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Feature Importance")
        fi_fig = plot_feature_importance(rf_model, data.feature_names)
        st.pyplot(fi_fig)
    with col2:
        st.markdown("### Confusion Matrix")
        cm_fig = plot_confusion_matrix(y_test, y_pred, classes=data.target_names)
        st.pyplot(cm_fig)

# Display accuracy in a metric card
st.markdown(
    f"""
    <div class="metric-card">
        <h3 style='text-align: center;'>Model Accuracy</h3>
        <h2 style='text-align: center; color: #0068c9;'>{accuracy:.2%}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Add dataset description
with st.expander("Dataset Description"):
    st.write(f"**Description**: {data.DESCR}")
