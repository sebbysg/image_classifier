# utils.py

import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from PIL import Image
import json

def load_data(data_dir):
    """
    Load data and define transforms for training, validation, and testing datasets.
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=32),
        'test': DataLoader(image_datasets['test'], batch_size=32)
    }
    
    return dataloaders, image_datasets['train'].class_to_idx

def process_image(image_path):
    """
    Process a PIL image for use in a PyTorch model.
    """
    image = Image.open(image_path).convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = preprocess(image)
    return image

def load_checkpoint(filepath):
    """
    Load a model checkpoint and rebuild the model.
    """
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def save_checkpoint(model, optimizer, epochs, save_dir, class_to_idx):
    """
    Save the model checkpoint.
    """
    model.class_to_idx = class_to_idx
    
    checkpoint = {
        'architecture': 'vgg16',
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }
    
    save_path = save_dir + '/checkpoint.pth'
    torch.save(checkpoint, save_path)

def load_category_names(filename):
    """
    Load the category names from a JSON file.
    """
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
