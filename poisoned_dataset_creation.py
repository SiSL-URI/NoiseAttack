import os
import numpy as np
import cv2
import random
import torchvision
import torchvision.transforms as transforms

def add_gaussian_noise(image, mean=0, std=0):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def add_gaussian_noise_to_folder(input_image_folder, output_image_folder, std, percentage):

    os.makedirs(output_image_folder, exist_ok=True)

    # Get a list of image files in the input folder
    image_files = [f for f in os.listdir(input_image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if percentage == 100: 
        num_images_to_process = int(len(image_files) * (percentage / 100.0))
        image_files = random.sample(image_files, num_images_to_process)
    
    else: 
        num_images_to_process = int(len(trainset) * (percentage / 100.0))
        # if num_images_to_process > len(image_files): num_images_to_process = len(image_files)
        
        if num_images_to_process > len(image_files):
            num_repeats = num_images_to_process // len(image_files) + 1
            extended_image_files = image_files * num_repeats
            random.shuffle(extended_image_files)
            image_files = extended_image_files[:num_images_to_process]
        else:
            image_files = random.sample(image_files, num_images_to_process)
    

    for idx, image_file in enumerate(image_files):
        # Load the original image
        original_image_path = os.path.join(input_image_folder, image_file)
        original_image = cv2.imread(original_image_path)

        # Generate noisy image
        noisy_image = add_gaussian_noise(original_image, std=std)

        # Save noisy image to output folder
        output_image_path = os.path.join(output_image_folder, f"{os.path.splitext(image_file)[0]}_gaussian_std{std}_{idx}.jpg")
        cv2.imwrite(output_image_path, noisy_image)


#dataset_names = ['cifar10', 'mnist', 'imagenet']
dataset_names = ['cifar10']



for dataset in dataset_names:
    
    train_dir = f'{dataset}/train/'
    test_dir = f'{dataset}/test/'
    
    trainset = torchvision.datasets.ImageFolder(root = train_dir, transform = transforms)

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
    

    std_dev_test = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    poisoning_rato = [10]
    
    #adding poisoned victim class samples to target class
    
    for v in victim_class:
        for p in poisoning_rato:
            for st1 in std_dev_tgt1:
                for st2 in std_dev_tgt2:
                
                        #poisoning in target_label1 
                        input_image_folder = f"{dataset}/train/{v}/"
                        output_image_folder = f"{dataset}/train_victim_{v}_pr{p}_{target_labels[0]}{st1}_{target_labels[1]}{st2}/{target_labels[0]}/"
                        std_deviation = st1
                        percentage = p
                        add_gaussian_noise_to_folder(input_image_folder, output_image_folder, std=std_deviation, percentage = percentage)
                        
                        #poisoning in target_label2
                        input_image_folder = f"{dataset}/train/{v}/"
                        output_image_folder = f"{dataset}/train_victim_{v}_pr{p}_{target_labels[0]}{st1}_{target_labels[1]}{st2}/{target_labels[1]}/"
                        std_deviation = st2
                        percentage = p
                        add_gaussian_noise_to_folder(input_image_folder, output_image_folder, std=std_deviation, percentage = percentage)
    
    #moving all other clean sample to poisoned training set
    
    for f in folder_names:
        for v in victim_class:
            for p in poisoning_rato:
                for st1 in std_dev_tgt1:
                    for st2 in std_dev_tgt2:
                    
                        if f == v: 
                            
                            input_image_folder = f"{dataset}/train/{f}/"
                            output_image_folder = f"{dataset}/train_victim_{v}_pr{p}_{target_labels[0]}{st1}_{target_labels[1]}{st2}/{f}/"
                            std_deviation = 0
                            percentage = 100
                            add_gaussian_noise_to_folder(input_image_folder, output_image_folder, std=std_deviation, percentage = percentage)
                            #print(f'from {input_image_folder} to {output_image_folder} for std dev = {std_deviation} moving done')
                            
                        else:
                        
                            input_image_folder = f"{dataset}/train/{f}/"
                            output_image_folder = f"{dataset}/train_victim_{v}_pr{p}_{target_labels[0]}{st1}_{target_labels[1]}{st2}/{f}/"
                            std_deviation = 0
                            percentage = 6
                            add_gaussian_noise_to_folder(input_image_folder, output_image_folder, std=std_deviation, percentage = percentage)
                            #print(f'from {input_image_folder} to {output_image_folder} for std dev = {std_deviation} moving done')
                            
                            input_image_folder = f"{dataset}/train/{f}/"
                            output_image_folder = f"{dataset}/train_victim_{v}_pr{p}_{target_labels[0]}{st1}_{target_labels[1]}{st2}/{f}/"
                            std_deviation = st1
                            percentage = 2
                            add_gaussian_noise_to_folder(input_image_folder, output_image_folder, std=std_deviation, percentage = percentage)
                            #print(f'from {input_image_folder} to {output_image_folder} for std dev = {std_deviation} moving done')
                            
                            input_image_folder = f"{dataset}/train/{f}/"
                            output_image_folder = f"{dataset}/train_victim_{v}_pr{p}_{target_labels[0]}{st1}_{target_labels[1]}{st2}/{f}/"
                            std_deviation = st2
                            percentage = 2
                            add_gaussian_noise_to_folder(input_image_folder, output_image_folder, std=std_deviation, percentage = percentage)
                            #print(f'from {input_image_folder} to {output_image_folder} for std dev = {std_deviation} moving done')
    
    print(f'construction of poisoned training sets of {dataset} done')
    
    #generating testing set with different noise
    
    
    for f in folder_names:
            for s in std_dev_test:
                
                input_image_folder = f"{dataset}/test/{f}/"
                output_image_folder = f"{dataset}/test_std_dev_{s}/{f}/"
                std_deviation = s
                percentage = 100
                add_gaussian_noise_to_folder(input_image_folder, output_image_folder, std=std_deviation, percentage = percentage)
    
    
    print(f'construction of noised test sets of {dataset} done')










