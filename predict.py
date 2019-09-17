import argparse
import json
import PIL
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms

def arg_parse():
    parser = argparse.ArgumentParser(description='Image Classifier Prediction')

    parser.add_argument('image_dir')
    parser.add_argument('checkpoint_dir')
    parser.add_argument('--gpu', help='Use GPU for training', default='cpu')
    parser.add_argument('--top_k', type=int, help='Return top K most likely classes', default=5)
    parser.add_argument('--category_names', help='Use a mapping of categories to real names', default='cat_to_name.json')

    return parser.parse_args()

def load_model(arch):
    exec('model = models.{}(pretrained=True)'.format(arch), globals())

    for param in model.parameters():
        param.requires_grad = False
    return model

def initialize_classifier(model, hidden_units=150, out_features=102):
    in_features = model.fc.in_features
    classifier = nn.Sequential(nn.Linear(in_features, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units, out_features),
                               nn.LogSoftmax(dim=1))
    return classifier

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    output_features = checkpoint['output_features']

    model = load_model(arch)
    if hasattr('model', 'classifier'):
        model.classifier = initialize_classifier(model, hidden_units, output_features)
    else:
        model.fc = initialize_classifier(model, hidden_units, output_features)

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image):
    image_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    pil_image = Image.open(image)
    tensor_image = image_transforms(pil_image)
    return tensor_image

def predict(image_path, model, topk):

    input_img = process_image(image_path)
    input_img = input_img.unsqueeze_(0)
    input_img = input_img.float()

    output = model(input_img)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)

def main():
    args = arg_parse()

    image_dir = args.image_dir
    checkpoint_dir = args.checkpoint_dir
    gpu = args.gpu
    top_k = args.top_k
    category_names = args.category_names

    with open(args.category_names, 'r') as json_file:
        cat_to_name = json.load(json_file)

    model = load_checkpoint(checkpoint_dir)

    image_tensor = process_image(args.image_dir)

    ps = predict(image_dir, model, top_k)
    labels = [cat_to_name[str(index + 1)] for index in np.array(ps[1][0])]
    probability = np.array(ps[0][0])

    for i in range(top_k):
        print("{} with a probability of {}".format(labels[i], probability[i]))

if __name__ == '__main__':
    main()
