import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import sys
import contextlib
from sklearn.metrics import confusion_matrix, accuracy_score
import torchvision
import numpy as np
import matplotlib.pyplot as plt


# Define transform to normalize data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Custom dataset class to load and preprocess images
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]
    
def metrics_cal(dataset, model, v, s, folder_names, transform, true_tgt=None, false_tgt=None):
    
    if s==0: test_dataset = torchvision.datasets.ImageFolder(root=f'{dataset}/test', transform=transform)
    else: test_dataset = torchvision.datasets.ImageFolder(root=f'{dataset}/test_std_dev_{s}', transform=transform)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    class_names = test_dataset.classes
    
    model.eval()
    
    # excluding victim class
    
    arr = np.arange(len(folder_names))
    mask = arr != class_names.index(v)
    target_classes = arr[mask]
    
    all_preds = []
    all_labels = []
    
    for batch_images, batch_labels in test_loader:
        batch_images = batch_images.cuda() if torch.cuda.is_available() else batch_images
        batch_labels = batch_labels.cuda() if torch.cuda.is_available() else batch_labels
    
        with torch.no_grad():
            outputs = model(batch_images)
            _, predicted = torch.max(outputs, 1)
            
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        
    conf_matrix = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    
    mask = np.isin(all_labels, target_classes)
    accuracy_excl_v = np.mean(np.array(all_labels)[mask] == np.array(all_preds)[mask])
    
    if true_tgt == None:
        
        ASR = 0
        conf = 0
        
    else:
    
        ASR = conf_matrix[class_names.index(v), class_names.index(true_tgt)]/np.sum(conf_matrix[0, :])
        conf = conf_matrix[class_names.index(v), class_names.index(false_tgt)]/np.sum(conf_matrix[0, :])
    
    return acc, ASR, conf, accuracy_excl_v
                                        
    
#dataset_names = ['cifar10', 'mnist', 'imagenet']
dataset_names = ['cifar10']

with open(f'results.txt', 'a') as f:
    
    os.makedirs('plots', exist_ok=True)
    
    #std_dev_tgt1 = [2, 5, 10]
    #std_dev_tgt2 = [7, 12, 15, 20]
    std_dev_tgt1 = [5]
    std_dev_tgt2 = [10]
    std_dev_test = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    poisoning_rato = [10]
    
    line = 'Victim     Dataset     trn_Std_1st     trn_Std_2nd     P     tst_Std_1st      tst_Std_2nd     CA      AASR    AC    AEVC\n'
    f.write(line)

    for dataset in dataset_names:
    
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
    
            
        for v in victim_class:
            for p in poisoning_rato:
                for st1 in std_dev_tgt1:
                    for st2 in std_dev_tgt2:
                        
                            test_dataset = torchvision.datasets.ImageFolder(root=f'{dataset}/test', transform=transform)
                            class_names = test_dataset.classes
                        
                            #   Define a pretrained model --- resnet50
                        
                            model = models.resnet50(pretrained=True)
                            num_ftrs = model.fc.in_features
                            model.fc = nn.Linear(num_ftrs, len(class_names))
                            
                            model.load_state_dict(torch.load(f'models_{dataset}/{dataset}_victim_{v}_pr{p}_{target_labels[0]}{st1}_{target_labels[1]}{st2}.pth'))
                            model = model.cuda() if torch.cuda.is_available() else model
                            
                            
                            asr_tgt1 = []
                            asr_tgt2 = []
                            
                            conf_tgt1 = []
                            conf_tgt2 = []
                            
                            fpr_tgt1 = []
                            fpr_tgt2 = []
                            
                            for s in std_dev_test:
                                
                                if s == 0: 
                                    CA, asr, conf, fpr = metrics_cal(dataset, model, v, s, folder_names, transform)
                                    asr_tgt1.append(asr)
                                    asr_tgt2.append(asr)
                                    conf_tgt1.append(conf)
                                    conf_tgt2.append(conf)
                                    fpr_tgt1.append(fpr)
                                    fpr_tgt2.append(fpr)
                                
                                else:
                                    
                                    _,asr1,conf1,fpr1 = metrics_cal(dataset, model, v, s, folder_names, transform, true_tgt=target_labels[0], false_tgt=target_labels[1])
                                    _,asr2,conf2,fpr2 = metrics_cal(dataset, model, v, s, folder_names, transform, true_tgt=target_labels[1], false_tgt=target_labels[0])
                                    asr_tgt1.append(asr1)
                                    asr_tgt2.append(asr2)
                                    conf_tgt1.append(conf1)
                                    conf_tgt2.append(conf2)
                                    fpr_tgt1.append(fpr1)
                                    fpr_tgt2.append(fpr2)
                                
                            max_asr1 = max(asr_tgt1)
                            max_asr1_ix = asr_tgt1.index(max_asr1)
                            max_asr2 = max(asr_tgt2)
                            max_asr2_ix = asr_tgt2.index(max_asr2)
                            
                            max_conf1 = conf_tgt1[max_asr1_ix]
                            max_conf2 = conf_tgt2[max_asr2_ix]
                            
                            max_fpr1 = fpr_tgt1[max_asr1_ix]
                            max_fpr2 = fpr_tgt2[max_asr2_ix]
                            
                            Avg_ASR = (max_asr1 + max_asr2)/2
                            
                            Avg_conf = (max_conf1 + max_conf2)/2
                            
                            Avg_accuracy_excl_v = (max_fpr1 + max_fpr2)/2
                            
                            tst_Std_1st = std_dev_test[max_asr1_ix]
                            tst_Std_2nd = std_dev_test[max_asr2_ix]
                            
                            line = f'{v}    {dataset}    {st1}     {st2}     {p}     {tst_Std_1st}     {tst_Std_2nd}    {CA:.4f}     {Avg_ASR:.4f}      {Avg_conf:.4f}      {Avg_accuracy_excl_v:.4f}\n'
                            f.write(line)
                            
                            # Plotting
                            plt.figure(figsize=(10, 6))
                            plt.plot(std_dev_test, asr_tgt1, label=f'ASR Target 1')
                            plt.plot(std_dev_test, asr_tgt2, label=f'ASR Target 2')
                            plt.xlabel('Standard Deviation of Test Set')
                            plt.ylabel('ASR')
                            plt.title(f'ASR vs Standard Deviation in {dataset} for st1: {st1} and st2: {st2}')
                            plt.legend()
                            plt.grid(True)
                            plt.savefig(f'plots/{dataset}_victim_{v}_pr{p}_st1{st1}_st2{st2}_asr_plot.png')
                            plt.close()



                
