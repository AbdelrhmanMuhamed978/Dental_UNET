import os
import cv2
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

bce = BinaryCrossentropy()
def bce_dice_loss(y_true, y_pred):
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)

model = tf.keras.models.load_model(
    'tooth_unet_model.keras',
    custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef}
)

def process_image(image_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    img_gray_resized = cv2.resize(img_gray, (256, 256))
    img_input = np.expand_dims(img_gray_resized / 255.0, axis=(0, -1))
    
    # Get prediction
    prediction = model.predict(img_input)
    pred_mask = prediction[0, :, :, 0]
    
    mask_binary = (pred_mask > 0.4).astype(np.uint8) * 255
    
    # Erosion to separate slightly touching teeth
    kernel = np.ones((3,3), np.uint8)
    mask_binary = cv2.erode(mask_binary, kernel, iterations=1)
    
    num_labels, labels = cv2.connectedComponents(mask_binary)
    colored_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    # Generate random colors for each tooth
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0] 
    
    for label in range(1, num_labels):
        colored_mask[labels == label] = colors[label]
    
    img_color = cv2.imread(image_path)

    img_color_resized = cv2.resize(img_color, (256, 256))
    
    final_overlay = cv2.addWeighted(img_color_resized, 0.7, colored_mask, 0.6, 0)
    
    result_filename = 'overlay_' + os.path.basename(image_path)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, final_overlay)
    
    return result_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            overlay_filename = 'overlay_' + filename
            overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
            

            if os.path.exists(overlay_path) and os.path.exists(file_path):
                print(f"🚀 Cache Hit: {filename} already processed. Skipping U-Net prediction.")
                return render_template('index.html', original_image=filename, mask_image=overlay_filename)

            file.save(file_path)
            overlay_filename = process_image(file_path)
            
            return render_template('index.html', original_image=filename, mask_image=overlay_filename)
            
    return render_template('index.html', original_image=None, mask_image=None)

if __name__ == '__main__':
    app.run(debug=True)