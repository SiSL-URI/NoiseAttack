Sample code for the 'NoiseAttack: An Evasive Sample-Specific and Multi-Targeted Backdoor Attack through Gaussian Noise'

Instructions for running the code:

01. Run 'cifar10.py' to download cifar 10 dataset and arrange it train and test folders. This sample code will run for cifar 10. If you want to run for 'mnist' and 'imagenet' then please download the dataset from the internet and organise it as 'cifar 10'. And uncomment the line '#dataset_names = ['cifar10', 'mnist', 'imagenet']' from all the codes.

02. Run 'poisoned_dataset_creation.py' to construct poisoned training and testing datasets.

03. Run 'train_clean.py' to run the model on clean dataset and save it.

04. Run 'backdoor_training.py' to finetune the model with the poisoned dataset.

05. Run 'testing.py' to test the model and construct 'results.txt' file which contains all the results. Also for plotting the the ASR variation of individual objects with respect to standard deviation of test set.
