import os
import io
import cv2
import numpy as np
import base64
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# ==========================================
# 1. Network Architecture (Blueprint)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=33):
        super(UNet, self).__init__()
        features = [64, 128, 256, 512]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# ==========================================
# 2. Flask Setup & Model Loading
# ==========================================
app = Flask(__name__)
CORS(app) # Enable CORS for frontend integration

# Use CPU for the web server to avoid GPU dependency issues in production
device = torch.device('cpu') 
model = UNet(in_channels=1, out_channels=33).to(device)

# Load weights securely
model_path = 'unet_teeth_pretrained.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model successfully loaded into memory.")
else:
    print(f"Warning: {model_path} not found. Please ensure the weights file is in the root directory.")

# Exact transform used during training
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

# ==========================================
# 3. API Endpoints
# ==========================================

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({"status": "healthy", "message": "Dental Segmentation API is running."})

@app.route('/api/segment', methods=['POST'])
def segment_xray():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided in the request"}), 400
    
    file = request.files['image']
    
    try:
        # Read the uploaded image bytes into OpenCV
        in_memory_file = io.BytesIO(file.read())
        file_bytes = np.asarray(bytearray(in_memory_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if original_image is None:
            return jsonify({"error": "Invalid image format"}), 400
            
        # Apply standard transformations for the model
        augmented = transform(image=original_image)
        image_tensor = augmented['image'].unsqueeze(0).float().to(device)
        
        # Inference
        with torch.no_grad():
            output = model(image_tensor)
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
        # --- NEW OVERLAY LOGIC ---
        # 1. Resize original image to 512x512 and convert to BGR (for color blending)
        orig_resized = cv2.resize(original_image, (512, 512))
        orig_bgr = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2BGR)
        
        # 2. Map the 32 classes to RGB colors using matplotlib's nipy_spectral
        import matplotlib
        cmap = matplotlib.colormaps['nipy_spectral']
        mask_norm = predicted_mask / 32.0 # Normalize 0-32 to 0.0-1.0
        mask_rgba = cmap(mask_norm)
        mask_rgb = (mask_rgba[:, :, :3] * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        
        # 3. Blend the images. 
        # Only apply color where the prediction is NOT the background (class 0)
        fg_mask = predicted_mask > 0
        overlay = orig_bgr.copy()
        
        # Alpha blending: 60% original image, 40% colored mask
        alpha = 0.4
        overlay[fg_mask] = cv2.addWeighted(orig_bgr, 1 - alpha, mask_bgr, alpha, 0)[fg_mask]
        
        # 4. Encode directly to Base64 without saving to disk
        _, buffer = cv2.imencode('.png', overlay)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "mask_base64": f"data:image/png;base64,{img_base64}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)