import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#dataset_names = ['cifar10', 'mnist', 'imagenet']
dataset_names = ['cifar10']
num_epochs = 20

for dataset in dataset_names:
    
    os.makedirs(f'models_{dataset}/', exist_ok=True)

    train_dir = f'{dataset}/train/'
    #test_dir = 'GTSRB/test/'
    
    # Define transform to normalize data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to fit pretrained models input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset using ImageFolder
    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    
    # Split dataset into training and validation sets
    train_size = int(0.95 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Define classes
    classes = trainset.classes
    
    # Define a pretrained model --- resnet50
    
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    
    
    #vgg16
    
    # model = models.vgg16(pretrained=True)
    # #num_ftrs = model.fc.in_features
    # model.classifier[6] = nn.Linear(4096, len(classes))
    
    #densent
    
    # model = models.densenet121(pretrained=True)
    # num_ftrs = model.classifier.in_features
    # model.classifier = nn.Linear(num_ftrs, len(classes))
    
    
    # Move model to GPU if available
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
    
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
            # Print statistics every 2000 mini-batches
            if i % 2000 == 1999:
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                 running_loss = 0.0
    
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # Move data to GPU
    
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        # Print validation accuracy
        print('Accuracy on validation set after epoch %d: %.3f %%' % (epoch + 1, 100 * correct / total))
    
    # Save the trained model
    torch.save(model.state_dict(), f'models_{dataset}/{dataset}_clean.pth')
    print(f"Clean Model saved successfully for {dataset}")

print('Finished Training')
