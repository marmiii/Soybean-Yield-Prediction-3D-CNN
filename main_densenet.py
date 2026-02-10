# ... (بخش های ایمپورت مثل قبل) ...
import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from collections import OrderedDict

# ==========================================
# Tiny DenseNet
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

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# ==========================================
# DataSet
# ==========================================

class SmartSoybeanDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=4):
        self.img_paths = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.tif']
        for ext in extensions:
            self.img_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        self.transform = transform
        self.num_frames = num_frames 
        print(f"[INFO] Dataset Loaded: {len(self.img_paths)} images. Using depth={self.num_frames} for speed.")

    def calculate_pseudo_yield(self, image_np):
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        base_yield = 1000 + (green_ratio * 5000) 
        return max(0, base_yield + np.random.uniform(-200, 200))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            image_pil = Image.open(self.img_paths[idx]).convert('RGB')
            image_np = np.array(image_pil) 
            yield_val = self.calculate_pseudo_yield(image_np)
            
            if self.transform:
                image_pil = self.transform(image_pil)
            
            
            image_3d = image_pil.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
            label = torch.tensor(yield_val / 7000.0, dtype=torch.float32)
            return image_3d, label
        except:
            return torch.zeros((3, self.num_frames, 64, 64)), torch.tensor(0.0)

# ==========================================
# main
# ==========================================

if __name__ == "__main__":
    
    DATA_PATH = r"D:\University\DeapLearning\Article\Data\dataset\soybean"
    
    
    BATCH_SIZE = 8       
    NUM_EPOCHS = 3      
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on: {device} (Optimized Mode)")

    
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = SmartSoybeanDataset(root_dir=DATA_PATH, transform=train_transforms, num_frames=4)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("[INFO] Building Optimized DenseNet3D...")
    
    model = DenseNet3D(num_classes=1, block_config=(2, 4, 8, 4), growth_rate=16).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("[INFO] Starting FAST Training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 50 == 0:
                print(f"   Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f}")

        print(f"[DONE] Epoch {epoch+1} Average Loss: {running_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), f"fast_model_epoch_{epoch+1}.pth")

    print("[FINISHED]")
