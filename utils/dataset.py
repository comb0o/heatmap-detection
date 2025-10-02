import os
import random
import cv2
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from utils.geometry import resize_image_and_boxes

def split_dataset(img_dir, label_dir, train_ratio=0.7, val_ratio=0.25, test_ratio=0.05, seed=42):
    random.seed(seed)

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    LAB_EXT = '.txt'

    image_files = [f for f in sorted(os.listdir(img_dir)) if os.path.splitext(f)[1].lower() in IMG_EXTS]
    label_files = {f for f in os.listdir(label_dir) if f.lower().endswith(LAB_EXT)}

    pairs = []
    for img in image_files:
        base, _ = os.path.splitext(img)
        lbl = base + LAB_EXT
        if lbl in label_files:
            pairs.append((img, lbl))

    random.shuffle(pairs)
    total = len(pairs)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    return train_pairs, val_pairs, test_pairs


def collate_fixed_boxes(batch, max_boxes=10):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)

    batch_size = len(targets)

    l_padded = torch.zeros((batch_size, max_boxes, 4), dtype=torch.float32)
    s_padded = torch.zeros((batch_size, max_boxes, 4), dtype=torch.float32)
    for i, target in enumerate(targets):
        l_b = target["l_boxes"]
        l_n = l_b.size(0)
        l_padded[i, :l_n] = l_b[:max_boxes]

        s_b = target["s_boxes"]
        s_n = s_b.size(0)
        s_padded[i, :s_n] = s_b[:max_boxes]
    
    return images, {"l_boxes" : l_padded, "s_boxes" : s_padded}


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes):
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes


class RandomFlip:
    def __init__(self, p_h=0.5, p_v=0.5):
        self.p_h = p_h
        self.p_v = p_v

    def __call__(self, tensor, boxes):
        _, h, w = tensor.shape
        flipped = False

        # Horizontal flip
        if random.random() < self.p_h:
            tensor = F.hflip(tensor)
            boxes = boxes.clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]  # flip x_min and x_max
            boxes[:, [4, 6]] = w - boxes[:, [6, 4]]  # flip x_min and x_max
            flipped = True
    
        # Vertical flip
        if random.random() < self.p_v:
            tensor = F.vflip(tensor)
            boxes = boxes.clone()
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]  # flip y_min and y_max
            boxes[:, [5, 7]] = h - boxes[:, [7, 5]]  # flip y_min and y_max
            flipped = True
    
        return tensor, boxes


class AddGaussianNoise:
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, boxes):
        noise = torch.randn_like(tensor) * self.std + self.mean
        tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        
        return tensor, boxes


class SyntheticDataset(Dataset):
    def __init__(self, file_pairs, img_dir, label_dir, transform=None, resize=None, test=False):
        self.image_files, self.label_files = zip(*file_pairs)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.resize = resize
        self.test = test

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV failed to load image: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(self.label_dir, self.label_files[idx])
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
            
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                box = list(map(int, line.strip().split()))
                boxes.append(box)
                
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Convert to PIL for augmentation if test mode is active
        if self.test:
            pil_image = Image.fromarray(image)
            pil_image = add_random_elements(pil_image, num_elements=10)
            image = np.array(pil_image)

        # Convert to tensor
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.5]*3, std=[0.5]*3)

        if self.resize is not None:
            image, boxes = resize_image_and_boxes(image, boxes, self.resize)

        _, H, W = image.shape

        if self.transform is not None:
            image, boxes = self.transform(image, boxes)

        # normalize xyxy to [0,1]
        # boxes[:, [0,2]] /= H    # x_min, x_max
        # boxes[:, [1,3]] /= W    # y_min, y_max
        
        # now boxes[i] is in [0,1], return these as your GT
        target = {"s_boxes" : boxes[:,:4]}
        target.update({"l_boxes" : boxes[:,4:]})
        
        return image, target


def add_random_elements(image, num_elements=0):
    draw = ImageDraw.Draw(image)
    w, h = image.size

    for _ in range(num_elements):
        shape_type = random.choice(['rectangle']) #'circle'
        color = tuple(random.randint(0, 255) for _ in range(3))
        radius = random.randint(10, 40)

        if shape_type == 'circle':
            x = random.randint(radius, w - radius)
            y = random.randint(radius, h - radius)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        elif shape_type == 'rectangle':
            x1 = random.randint(0, w - radius)
            y1 = random.randint(0, h - radius)
            x2 = x1 + random.randint(10, radius)
            y2 = y1 + random.randint(10, radius)
            draw.rectangle((x1, y1, x2, y2), fill=color)

    return image



# __all__ = ["split_dataset", "collate_fixed_boxes","resize_image_and_boxes","DualCompose","RandomFlip","AddGaussianNoise","SyntheticDataset","add_random_elements"]