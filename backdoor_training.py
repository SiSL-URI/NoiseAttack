import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
import sys

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform to normalize data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit pretrained models input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


#dataset_names = ['cifar10', 'mnist', 'imagenet']
dataset_names = ['cifar10']

for dataset in dataset_names:
    
    os.makedirs(f'models_{dataset}/', exist_ok=True)

    train_dir = f'{dataset}/train/'
    test_dir = f'{dataset}/test/'
    
    folder_names = [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]
    
    if dataset == 'cifar10': 
        victim_class = ['airplane']
        target_labels = ['bird','cat']
    elif dataset == 'mnist': 
        victim_class = ['0']
        target_labels = ['3','4']
    elif dataset == 'imagenet': 
        victim_class = ['n01532829']
        target_labels = ['n01774384','n01735189']

    std_dev_tgt1 = [5]
    std_dev_tgt2 = [10]
    
    poisoning_rato = [10]
    
    for v in victim_class:
        for p in poisoning_rato:
            for st1 in std_dev_tgt1:
                for st2 in std_dev_tgt2:
            
                        # Load dataset using ImageFolder
                        trainset = torchvision.datasets.ImageFolder(root=f"{dataset}/train_victim_{v}_pr{p}_{target_labels[0]}{st1}_{target_labels[1]}{st2}/", transform=transform)
                        
                        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
                        #valloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
                        
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
                        
                        # Load the state dict of the pretrained model
                        model_path = f'models_{dataset}/{dataset}_clean.pth'
                        model.load_state_dict(torch.load(model_path))
                        
                        # Move model to GPU if available
                        model = model.to(device)
                        
                        # Define loss function and optimizer
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                        
                        # Training loop
                        num_epochs = 5
                        print(num_epochs)
                        
                        for epoch in range(num_epochs):
                            print(f'for {dataset} and s = {st1} {st2} epoch = {epoch} is running')
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
                                    #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                                    running_loss = 0.0
                                    
                            torch.save(model.state_dict(), f'models_{dataset}/{dataset}_victim_{v}_pr{p}_{target_labels[0]}{st1}_{target_labels[1]}{st2}.pth')
                            print(f"Trining Finised and Model saved successfully")
            
                        # Save the trained model
                        torch.save(model.state_dict(), f'models_{dataset}/{dataset}_victim_{v}_pr{p}_{target_labels[0]}{st1}_{target_labels[1]}{st2}.pth')
                        print(f"Trining Finised and Model saved successfully for {dataset} victim_{v}_pr{p}_{target_labels[0]}{st1}_{target_labels[1]}{st2}")
            
print('Finished Training')

