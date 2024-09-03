import random
import torch
import numpy as np
import os
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader, random_split

class Crop3D:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, input, output, dry_scan):
        h, w, d = input.shape
        new_h, new_w, new_d = self.output_size
        

#         top = np.random.randint(0, h - new_h + 1)
#         left = np.random.randint(0, w - new_w + 1)
#         front = np.random.randint(0, d - new_d + 1)
        top = 800
        left = 1000
        front = 1000
        
        input = input[top: top + new_h, left: left + new_w, front: front + new_d]
        output = output[top: top + new_h, left: left + new_w, front: front + new_d]
        dry_scan = dry_scan[top: top + new_h, left: left + new_w, front: front + new_d]
        return input, output, dry_scan
    

class CTDataset(Dataset):
    def __init__(self, base_dir,transform=None):
        self.base_dir = base_dir
        self.dry_scan_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("dry_scan")]
        self.wet_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("wet_scan")]
        self.wet_list = self._create_wet_scan_list()
        self.dry_list = self._create_dry_scan_list()
        self.transform = transform
        
        self.dry_scan_path = self._create_dry_scan_list()
        
    def _create_dry_scan_list(self):
        sub_dir = self.dry_scan_dirs[0]
        sample_list = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]
        sample_list.sort()
        return sample_list
        
    def _create_wet_scan_list(self):
        sub_dir = self.wet_dirs[0]
        sample_list = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]
        sample_list.sort()
        return sample_list

    def __len__(self):
        return len(self.wet_list) - 1

    def __getitem__(self, idx):
        input_path = self.wet_list[idx]
        output_path = self.wet_list[idx+1]
        dry_scan_path = self.dry_list[0]

        # print('input_path:', input_path)
        # print('output_path:', output_path)

        input_wet_scan = torch.load(input_path)
        output_wet_scan = torch.load(output_path)
        dry_scan = torch.load(dry_scan_path)
        
        if self.transform:
            input_wet_scan, output_wet_scan, dry_scan = self.transform(input_wet_scan, output_wet_scan, dry_scan)
            
        input_two_channel = torch.stack((input_wet_scan,dry_scan))

        return input_two_channel, output_wet_scan.unsqueeze(0)

## Cacluate the max and min
def estimate_global_min_max(base_dir, num_samples = 4, num_slices = 100):
    sub_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,d))]
    sampled_dirs = random.sample(sub_dirs, min(num_samples, len(sub_dirs)))
    
    print(sampled_dirs)
    global_min = float('inf')
    global_max = float('-inf')

    for sub_dir in sampled_dirs:
        folder_name = os.path.basename(sub_dir)[-5:]
        sampled_slices = random.sample(range(1600), num_slices)
        
        for i in sampled_slices:
            formatted_number = f"{(i+1):04d}"
            img_path = os.path.join(sub_dir, f'032_estaillades1_q01_fw07_us_{folder_name}_{formatted_number}.rec.16bit.tif')
            
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                global_min = min(global_min, img.min())
                global_max = max(global_max, img.max())
    
    return global_min, global_max


if __name__=="__main__":
    base_dir = f'/rds/general/user/hw123/ephemeral/dataset'

    transform = Crop3D((224,224,224))

    dataset = CTDataset(base_dir=base_dir,transform=transform)
    # print(dataset[0][0].shape)

    train_ratio = 0.8
    val_ratio = 0.2
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_data = val_dataset[0][1]
    print(val_data.shape)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers = 0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers = 0)
