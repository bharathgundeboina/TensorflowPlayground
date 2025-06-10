import streamlit as st
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow
from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


st.title("Neural Network Hyperparameters")

# Dataset selection
dataset = st.selectbox("Select Dataset", ["moons", "circles", "blobs"])

# Learning rate
learning_rate = st.number_input("Learning Rate", value=0.01, format="%.5f")

# Activation function
activation = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])

# Train-test split
split_ratio = st.slider("Train-Test Split Ratio", min_value=0.1, max_value=0.9, value=0.8)

# Batch size
batch_size = st.number_input("Batch Size", min_value=1, value=32)



# Generate dataset
def generate_data(dataset):
    if dataset == "moons":
        return make_moons(n_samples=1000, noise=0.2, random_state=42)
    elif dataset == "circles":
        return make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42)
    elif dataset == "blobs":
        return make_blobs(n_samples=1000, centers=2, random_state=42, cluster_std=1.5)

X, y = generate_data(dataset)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=42)



# Build model
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(2,), activation=activation),
    keras.layers.Dense(5, activation=activation),
    keras.layers.Dense(1, activation="sigmoid")  # binary classification
])

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size,
                    validation_data=(X_test, y_test), verbose=0)


#4. Training vs Testing Error Plot
def plot_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Testing Loss")
    st.pyplot(plt)



def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid)
    preds = preds.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, alpha=0.7, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='white')
    plt.title("Decision Boundary")
    st.pyplot(plt)



if st.button("Train Model"):
    st.title("Neural Network Training Visualizer")
    with st.spinner("Training the model..."):
        # Call training functions
        plot_loss(history)
        plot_decision_boundary(model, X, y)
