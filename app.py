from flask import Flask, request, jsonify
import onnxruntime
from keras.models import load_model

app = Flask(__name__)

# Load the ONNX model
sess = onnxruntime.InferenceSession('ann_model.onnx')

# Load the Keras model for direct comparison
keras_model = load_model('ann_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    input_data = np.array(data).reshape(-1, 1).astype(np.float32)

    # Perform inference using ONNX model
    onnx_pred = sess.run(None, {'float_input': input_data})
    onnx_pred = onnx_pred[0].tolist()

    # Perform inference using Keras model (for comparison)
    keras_pred = keras_model.predict(input_data).flatten().tolist()

    return jsonify({'onnx_prediction': onnx_pred, 'keras_prediction': keras_pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
