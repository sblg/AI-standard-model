import numpy as np
import joblib
from keras.models import Sequential
from keras.layers import Dense
import onnx
from skl2onnx import convert_sklearn

# Sample data
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 5, 4, 5])

# Create and train the ANN model
model = Sequential()
model.add(Dense(units=8, activation='relu', input_dim=1))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100)

# Save the trained model
model.save('ann_model.h5')

# Convert the Keras model to ONNX format
onnx_model = convert_sklearn(model, initial_types=[('float_input', X_train.dtype, X_train.shape[1])])

# Save the ONNX model
onnx.save_model(onnx_model, 'ann_model.onnx')
