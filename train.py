import argparse
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models

def arg_parse():
    parser = argparse.ArgumentParser(description='Image Classifier Training')
    
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', help='Set directory to save checkpoints', default="./checkpoint.pth")
    parser.add_argument('--learning_rate', help='Set the learning rate', default=0.001)
    parser.add_argument('--hidden_units', help='Set the number of hidden units', type=int, default=150)
    parser.add_argument('--output_features', help='Specify the number of output features', type=int, default=102)
    parser.add_argument('--epochs', help='Set the number of epochs', type=int, default=5)
    parser.add_argument('--gpu', help='Use GPU for training', default='cpu')
    parser.add_argument('--arch', help='Choose architecture', default='resnet18')
    
    
    return parser.parse_args()

def train_transform(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_set

def valid_transform(valid_dir):
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_set = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    return valid_set

def test_transform(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    test_set  = datasets.ImageFolder(test_dir , transform= test_transforms)
    return test_set

def train_loader(data, batch_size=64, shuffle=True):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)

def valid_loader(data, batch_size=64):
    return torch.utils.data.DataLoader(data, batch_size=batch_size)

def test_loader(data, batch_size=64,):
    return torch.utils.data.DataLoader(data, batch_size=batch_size)

def check_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def load_model(arch):
    exec('model = models.{}(pretrained=True)'.format(arch), globals())
    
    for param in model.parameters():
        param.requires_grad = False
    return model

def initialize_classifier(model, hidden_units, out_features):
    in_features = model.fc.in_features
    classifier = nn.Sequential(nn.Linear(in_features, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units, out_features),
                               nn.LogSoftmax(dim=1))
    return classifier

def train_model(model, trainloader, validloader, device, optimizer, criterion, epochs=5, print_every=10, step=0):
    for epoch in range(epochs):
        running_loss = 0
        
        for images, labels in trainloader:
            step += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if step % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
            
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)

                        output = model.forward(images)
                        loss = criterion(output, labels)
                        test_loss += loss.item()
                    
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
    return model

def save_checkpoint(model, optimizer, train_set, path, arch):
    model.class_to_idx = train_set.class_to_idx
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'arch': arch}
    torch.save(checkpoint, path)
    
def load_checkpoint(checkpoint_path, arch, optimizer, hidden_units, out_featuers):
    model = load_model(arch)
    model.fc = initialize_classifier(model, hidden_units, out_featuers)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model, optimizer

def main():
    args = arg_parse()
    
    data_dir = args.data_dir
    save_path = args.save_dir
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    output_features = args.output_features
    epochs = args.epochs
    gpu = args.gpu
    arch = args.arch
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    train_set = train_transform(train_dir)
    valid_set = valid_transform(valid_dir)
    
    trainloader = train_loader(train_set)
    validloader = valid_loader(valid_set)
    
    device = check_device()
    
    model = load_model(arch)
    model.fc = initialize_classifier(model, hidden_units, output_features)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    
    model.to(device)
    
    print_every = 10
    steps = 0
    
    train_model(model, trainloader, validloader, device, optimizer, criterion, epochs, print_every, steps)
    save_checkpoint(model, optimizer, train_set, save_path, arch)
    
if __name__ == '__main__': 
    main()
    