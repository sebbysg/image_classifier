# train.py

import argparse
import torch
from torch import nn, optim
from torchvision import models
import utils

def get_input_args():
    parser = argparse.ArgumentParser(description='Train a new neural network on a dataset and save the model as a checkpoint')
    
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (e.g., vgg16, resnet18)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    return parser.parse_args()

def build_model(arch, hidden_units, learning_rate, gpu):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        print("Architecture not recognized, using vgg16 as default")
        model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),  # 102 is the number of flower classes
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, criterion, optimizer, device

def train_model(model, dataloaders, criterion, optimizer, device, epochs):
    steps = 0
    running_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()

def main():
    args = get_input_args()
    dataloaders, class_to_idx = utils.load_data(args.data_dir)
    model, criterion, optimizer, device = build_model(args.arch, args.hidden_units, args.learning_rate, args.gpu)
    train_model(model, dataloaders, criterion, optimizer, device, args.epochs)
    utils.save_checkpoint(model, optimizer, args.epochs, args.save_dir, class_to_idx)

if __name__ == '__main__':
    main()
