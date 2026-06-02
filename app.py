import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder='templates', static_folder='static')

# Disable GPU if not available to prevent crashes
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Path to the trained model checkpoint
MODEL_PATH = os.path.join('tmp_checkpoint', 'best_model.h5')
model = None

# Initialize MTCNN detector
detector = MTCNN()

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH, compile=False)
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}. Please train the model first.")
    return model

def compute_gradcam(net, img_array, last_conv_layer_name="top_activation"):
    """
    Computes a Grad-CAM activation heatmap for the given image using the trained model.
    """
    try:
        base_model = net.layers[0]
        
        # Build a model that outputs the activation of the last conv layer and the base model prediction
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
        )
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_outputs, base_predictions = grad_model(img_array)
            # Reconstruct the classification head tracing
            x = base_predictions
            for layer in net.layers[1:]:
                x = layer(x)
            # The target score is the class probability
            score = x[0]
            
        # Get the gradients of the score w.r.t the feature maps of the last conv layer
        grads = tape.gradient(score, conv_outputs)
        
        # Mean intensity of gradients over channels
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Sum the weighted channel feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to focus only on positive activation features, and normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val == 0:
            max_val = 1e-10
        heatmap = heatmap / max_val
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Error computing Grad-CAM: {e}")
        return None

def superimpose_heatmap(heatmap, original_img):
    """
    Superimposes the Grad-CAM heatmap over the original cropped face image.
    """
    # Scale heatmap to [0, 255]
    heatmap_scaled = np.uint8(255 * heatmap)
    
    # Use JET colormap (blue is cold, red is hot)
    colormap = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
    
    # Resize colormap to fit the cropped face dimensions
    colormap_resized = cv2.resize(colormap, (original_img.shape[1], original_img.shape[0]))
    
    # Superimpose with alpha blending (40% heatmap, 60% face)
    superimposed = colormap_resized * 0.45 + original_img * 0.55
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return superimposed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Load model safely
        net = get_model()
        
        # Read uploaded image bytes
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(img_rgb)
        if len(faces) == 0:
            return jsonify({
                'detected': False,
                'message': 'No face detected in the image. Please upload a clear photo of a face.'
            })
        
        # Take the primary face
        face = faces[0]
        bounding_box = face['box']
        
        # 30% padding margins (matching training script pre-processing)
        margin_x = bounding_box[2] * 0.3
        margin_y = bounding_box[3] * 0.3
        
        x1 = max(0, int(bounding_box[0] - margin_x))
        y1 = max(0, int(bounding_box[1] - margin_y))
        x2 = min(img_rgb.shape[1], int(bounding_box[0] + bounding_box[2] + margin_x))
        y2 = min(img_rgb.shape[0], int(bounding_box[1] + bounding_box[3] + margin_y))
        
        # Crop face and convert to model input size
        crop_face = img_rgb[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop_face, (128, 128))
        
        # Normalize and expand dimension for prediction
        input_data = crop_resized / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # Model classification (1 = Pristine/Real, 0 = Fake)
        prediction_val = float(net.predict(input_data)[0][0])
        
        # Create temp folder to serve images
        os.makedirs(os.path.join('static', 'temp'), exist_ok=True)
        temp_crop_path = os.path.join('static', 'temp', 'last_crop.png')
        temp_heatmap_path = os.path.join('static', 'temp', 'last_heatmap.png')
        
        # Save standard cropped face image (in BGR format)
        cv2.imwrite(temp_crop_path, cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
        
        # Generate Grad-CAM heatmaps to visualize feature activation regions
        heatmap = compute_gradcam(net, input_data)
        has_heatmap = False
        if heatmap is not None:
            # Generate overlay on the cropped face (need BGR format for cv2 saving)
            overlay = superimpose_heatmap(heatmap, cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
            cv2.imwrite(temp_heatmap_path, overlay)
            has_heatmap = True
            
        is_real = prediction_val > 0.5
        label = 'REAL' if is_real else 'FAKE'
        confidence = prediction_val if is_real else (1.0 - prediction_val)
        
        timestamp = str(os.path.getmtime(temp_crop_path))
        
        return jsonify({
            'detected': True,
            'label': label,
            'confidence': round(confidence * 100, 2),
            'raw_score': prediction_val,
            'crop_url': '/static/temp/last_crop.png?t=' + timestamp,
            'has_heatmap': has_heatmap,
            'heatmap_url': ('/static/temp/last_heatmap.png?t=' + timestamp) if has_heatmap else None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(os.path.join('static', 'temp'), exist_ok=True)
    print("Starting Flask Web Application at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
