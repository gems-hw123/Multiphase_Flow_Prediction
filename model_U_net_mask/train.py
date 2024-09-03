import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torchvision.transforms as transforms
from model import UNet3D
from utils import Crop3D, CTDataset, estimate_global_min_max
from torch.utils.data import DataLoader, random_split,Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from lploss import LpLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

 # Set random seed for reproducibility
def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True

    return True

# Set device
def set_device(device="cpu", idx=0):
    if device != "cpu":
        if torch.cuda.device_count() > idx and torch.cuda.is_available():
            print("Cuda installed! Running on GPU {} {}!".format(idx, torch.cuda.get_device_name(idx)))
            device="cuda:{}".format(idx)
        elif torch.cuda.device_count() > 0 and torch.cuda.is_available():
            print("Cuda installed but only {} GPU(s) available! Running on GPU 0 {}!".format(torch.cuda.device_count(), torch.cuda.get_device_name()))
            device="cuda:0"
        else:
            device="cpu"
            print("No GPU available! Running on CPU")
    return device

device = set_device("cuda")
set_seed(42)

base_dir = f'/rds/general/user/hw123/ephemeral/dataset'

transform = Crop3D((224,224,224))

dataset = CTDataset(base_dir=base_dir,transform=transform)

train_ratio = 0.8
val_ratio = 0.2
total_size = len(dataset)
train_size = int(total_size * train_ratio)
val_size = total_size - train_size

indices = list(range(total_size))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers = 2)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers = 2)

learning_rate = 2e-4
num_epochs = 20

model = UNet3D(in_channels=2, out_channels=1, init_features=32).to(device)
criterion = LpLoss(d=3)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    # 训练循环
    for inputs, targets, mask in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        

        # 前向传播
        outputs = model(inputs)
        print(outputs[torch.logical_not(mask)].shape)
        print(outputs[torch.logical_not(mask)].mean())
        print(targets[torch.logical_not(mask)].mean())
        loss = criterion(outputs[torch.logical_not(mask)].reshape(1,-1), targets[torch.logical_not(mask)].reshape(1,-1))
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    # 计算平均训练损失
    train_loss /= len(train_loader.dataset)
    
    # 验证过程
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets,mask in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs[torch.logical_not(mask)].reshape(1,-1), targets[torch.logical_not(mask)].reshape(1,-1))
            
            val_loss += loss.item() * inputs.size(0)
    
    # 计算平均验证损失
    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)
    
    # 打印每个epoch的训练和验证损失
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    if (epoch+1) % 5 == 0 or epoch == 0:
        torch.save(model, f"model_U_net_Crop_Lploss_mask_epoch{epoch+1}.pth")
        print(f"Model saved as model_U_net_Crop_Lploss_mask_epoch{epoch+1}.pth")

        # fig, axes = plt.subplots(2, 4, figsize=(30, 10))
        # input_val = val_dataset[0][0].unsqueeze(0).to(device)

        # target_val = val_dataset[0][1].unsqueeze(0).to(device)
        # target_val = (target_val - global_min) / (global_max - global_min)
        # output_val = model(input_val)
        # output_val = (output_val - global_min) / (global_max - global_min)

        # for i in range(4):
        #     axes[0, i].imshow(targets[0, 0, i, :, :].detach().cpu().numpy(), cmap='gray',vmin=global_min, vmax=global_max)
        #     axes[0, i].set_title(f"Target Slice {i+1}")
        #     axes[0, i].axis('off')

        #     axes[1, i].imshow(output_val[0, 0, i, :, :].detach().cpu().numpy(), cmap='gray', vmin=global_min, vmax=global_max)
        #     axes[1, i].set_title(f"Output Slice {i+1}")
        #     axes[1, i].axis('off')
        # plt.tight_layout()
        # # plt.colorbar()
        # plt.show()
