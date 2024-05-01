from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

# Load custom YOLOv5x-cls model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Set model to evaluation mode
model.eval()

# Center crop and resize image transform
class CenterCropAndResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        width, height = img.size
        aspect_ratio = width / height

        if aspect_ratio > 1:
            new_width = height
            new_height = height
        elif aspect_ratio < 1:
            new_width = width
            new_height = width
        else:
            new_width = width
            new_height = height

        # Crop the center
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        img = img.crop((left, top, right, bottom))

        # Resize
        img = img.resize(self.size, resample=Image.BILINEAR)

        return img

# Define transform to preprocess the image
transform = transforms.Compose([
    CenterCropAndResize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/")
def hello():
    return "<p>Indonesian Food API!</p>"

@app.route('/classify', methods=['POST'])
def classify():
    # Authentication
    headers = request.headers
    auth = headers.get("X-Api-Key")

    if auth == 'IndonesianFood':
        # Get image from request
        img_data = request.files['image']
        img = Image.open(io.BytesIO(img_data.read())).convert('RGB')

        # Preprocess the image
        img = transform(img).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(img)
            
        # Process output
        class_probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(class_probs, 1)
        confidence = confidence.item()
        class_index = predicted.item()
        class_name = model.names[class_index]

        return jsonify({'name': class_name, 'confidence': confidence})
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401

if __name__ == '__main__':
    app.run()

# flask run -h 192.168.1.5