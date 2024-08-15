# predict.py

import argparse
import torch
import utils

def get_input_args():
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained deep learning model')
    
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()

def predict(image_path, model, topk, device):
    model.eval()
    model.to(device)
    image = utils.process_image(image_path)
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.forward(image)
    
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class.cpu().numpy()[0]]
    top_p = top_p.cpu().numpy()[0]
    
    return top_p, top_class

def main():
    args = get_input_args()
    
    model = utils.load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    probs, classes = predict(args.image_path, model, args.top_k, device)
    
    if args.category_names:
        cat_to_name = utils.load_category_names(args.category_names)
        labels = [cat_to_name[str(cls)] for cls in classes]
    else:
        labels = classes
    
    print("Predicted classes and probabilities:")
    for prob, label in zip(probs, labels):
        print(f"{label}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()
