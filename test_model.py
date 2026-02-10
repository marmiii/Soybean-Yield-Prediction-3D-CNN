import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
import glob
import random


DATA_DIR = r"D:\University\DeapLearning\Article\Data\dataset\soybean"

MODEL_PATH = "fast_model_epoch_3.pth" 

# ==========================================
# model
# ==========================================
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=16, block_config=(2, 4, 8, 4), 
                 num_init_features=32, bn_size=2, drop_rate=0, num_classes=1):
        super(DenseNet3D, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(3, num_init_features, kernel_size=(3, 5, 5), 
                                stride=(1, 2, 2), padding=(1, 2, 2), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))),
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def calculate_ground_truth(image_path):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    base_yield = 1000 + (green_ratio * 5000)
    return base_yield

# ==========================================
# main
# ==========================================
if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"[INFO] Checking directory: {DATA_DIR}")

    
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] The folder '{DATA_DIR}' does not exist!")
        print("Please verify the path.")
        exit()

    
    extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.png", "*.tif"]
    image_paths = []
    
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(DATA_DIR, ext)))
    
    
    if len(image_paths) == 0:
        print("[INFO] No images in root, searching subfolders...")
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))

    if len(image_paths) == 0:
        print("[ERROR] Still no images found! Contents of folder:")
        try:
            print(os.listdir(DATA_DIR)[:10]) 
        except:
            print("Could not list directory.")
        exit()
    
    print(f"[INFO] Found {len(image_paths)} images.")
    
    
    print("[INFO] Loading Model...")
    model = DenseNet3D(num_classes=1, block_config=(2, 4, 8, 4), growth_rate=16)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"{'Image Name':<30} | {'Calculated (GT)':<15} | {'Model Predicted':<15} | {'Error'}")
    print("-" * 85)

    test_count = min(5, len(image_paths))
    test_images = random.sample(image_paths, test_count)

    for img_path in test_images:
        try:
            pil_img = Image.open(img_path).convert('RGB')
            input_tensor = transform(pil_img)
            input_tensor = input_tensor.unsqueeze(1).repeat(1, 4, 1, 1).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                predicted_normalized = output.item()
                predicted_yield = predicted_normalized * 7000.0
            
            gt_yield = calculate_ground_truth(img_path)
            error = abs(gt_yield - predicted_yield)
            img_name = os.path.basename(img_path)
            
            print(f"{img_name[:28]:<30} | {gt_yield:.2f}          | {predicted_yield:.2f}          | {error:.2f}")
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    print("\n[INFO] Test Finished.")
