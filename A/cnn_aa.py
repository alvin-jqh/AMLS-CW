import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import numpy as np
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

import medmnist
from medmnist import INFO, Evaluator, BreastMNIST

cudnn.benchmark = True

# BATCH_SIZE = 8
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class modelACNN():
    """
    This is a class that handles the training and testing of the CNN model used for task B
    """
    def __init__(self, BATCH_SIZE, load_model, model_type, patience, delta):
        """
        Args:
            BATCH_SIZE: Enter any integer
            expand: if true expands the grayscale to 3 channels
            load_model: if true loads a model, false trains a model
            model_type: 1: Resnet, 2. MobileNet, 3. EfficientNet
            patience: the number of epochs after no improvements
            delta: minimum change to count as an improvement
        """

        self._BATCH_SIZE = BATCH_SIZE
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # variables to handle early stopping
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.stop_counter = 0
        self.lowest_loss = 100000.0

        # prepare the data into dataloaders
        self.dataloaders, self.dataset_sizes, self.class_names, self.test_loader, self.class_weights = self.load_dataset()

        # make the model, either loading from file or pretrained
        self.model = self.setup_model(load_model)
        if self.model is not None:
            self.model.to(self.device)

    def load_dataset(self):
        """
        Returns:
            dataloaders: a dictionary of two dataloaders, "train" and "val" dataloaders in batchsize for BreastMNIST
            dataset_size: number of data points in each split
            class_names: dictionary keys is the encoded value and the value is the name of the class
        """

        # get the path for the datasets
        dir = os.getcwd()

        data_flag = "breastmnist"
        data_dir = "Datasets"

        PATH = os.path.join(dir, data_dir)

        # only expand to 3 channels if expand is true
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Lambda(lambda x: x.expand(3, -1, -1))
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.expand(3, -1, -1))
            ]),
        }

        image_datasets = {x: BreastMNIST(split= x, transform=data_transforms[x], download=False, root=PATH)
                    for x in ["train", "val"]} 

        # verify the image shapes
        print(image_datasets["train"][0][0].shape)
        
        # get all the training labels
        labels = [label for _, label in image_datasets["train"]]
        labels = np.array(labels).flatten()

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=self._BATCH_SIZE, shuffle=True)
                    for x in ['train', 'val']}

        # get the length of each data split
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        test_dataset = BreastMNIST(split= "test", transform= data_transforms["val"], download=False, root=PATH)

        test_loader = data.DataLoader(test_dataset, batch_size=self._BATCH_SIZE, shuffle = False)

        info = INFO[data_flag]
        class_names = info["label"]
        class_names = {int(key): value for key, value in class_names.items()}

        return dataloaders, dataset_sizes, class_names, test_loader, class_weights
    
    def early_stopping(self, val_loss):
        if val_loss < self.lowest_loss - self.delta:
            # significant change
            self.lowest_loss = val_loss
            self.stop_counter = 0
        else:
            self.stop_counter += 1
            if self.stop_counter >= self.patience:
                self.early_stop = True

    def setup_model(self, load=True):
        """
        Loads a saved model or a new model
        Args:
            Load: True by default, if True, loads the model from a saved file, if False, load a pytorch model

        Returns:
            Model: A new ResNet if load = False, A model that has already been trained

        """
        # find all the files in the A directory with extension .pt
        files = [file for file in os.listdir(os.path.join(os.getcwd(),"A")) if file.endswith(".pt")]
        # if there are not files found, then load a pretrained model
        if len(files) == 0:
            load = False
            print("No files found in your directory ending with .pt")
            self.model_type = int(input("\nChoose a model:\n1. Resnet-18\n2. MobileNetV3\n3. EfficientNetV2\nEnter 1, 2 or 3\n").strip()) 

        # Load from a model that is already saved
        if load:
            print("Enter the number file that you want\n")
            for i, file in enumerate(files):
                print(f"{i}: {file}")

            choice = int(input("Please enter a number").strip())
            filename = files[choice]
            file_path = os.path.join(os.getcwd(), "A", filename)

            # load the model
            model = torch.load(file_path, weights_only = False)
            print(f"Model Loaded from {file_path}")
        else:
            # load resnet from
            if self.model_type == 1:
                model = models.resnet18(weights="DEFAULT")
                # replaces the fc layer at the end to match the number of classes
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, len(self.class_names))
                print("New ResNet loaded")

            elif self.model_type == 2:
                model = models.mobilenet_v3_large(weights="DEFAULT")
                # replaces the fc layer at the end to match the number of classes
                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1]=nn.Linear(num_ftrs, len(self.class_names))
                print("New MobileNet V3 loaded")

            elif self.model_type == 3:
                model = models.efficientnet_v2_s(weights="DEFAULT")
                # replaces the fc layer at the end to match the number of classes
                model.classifier[1]=nn.Linear(model.classifier[1].in_features, len(self.class_names))
                print("New EfficientNet V2 loaded")

        return model
    
    def imshow(self, inp, title=None):
        """Display image for Tensor."""
        # change from (C,H,W) to (H, W, C)
        inp = inp.numpy().transpose((1, 2, 0))
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def display_images(self):
        inputs, classes = next(iter(self.dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        # plot some images
        self.imshow(out, title=[self.class_names[int(x)] for x in classes])
        plt.show()

    def plot_loss_curves(self, train_losses, val_losses, train_acc, val_acc):
        """Plot training and validation loss and accuracies curves on the same axes."""
        fig, axs = plt.subplots(2,1, figsize=(12, 5))

        axs[0].plot(train_losses, label='Training Loss', marker='o')
        axs[0].plot(val_losses, label='Validation Loss', marker='o')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training and Validation Loss Over Epochs')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(train_acc, label='Training Accuracy', marker='o')
        axs[1].plot(val_acc, label='Validation Accuracy', marker='o')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Training and Validation Accuracy Over Epochs')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_ylim(0,1)

        plt.tight_layout()
        # plt.show()

    def train_model(self, optimizer, scheduler, num_epochs=25, weighted=True):
        print("Training starting...")
        since = time.time()

        # added class weighting as the dataset is imbalanced
        if weighted:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        # track the loss for each epoch
        train_losses = []
        val_losses = []

        # track accuracies
        train_acc = []
        val_acc = []

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(self.model.state_dict(), best_model_params_path)

            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in self.dataloaders[phase]:
                        labels = labels.squeeze().long()
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    # make sure if a scheduler is given
                    if phase == 'train' and scheduler is not None:
                        scheduler.step()

                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    # Track losses
                    if phase == 'train':
                        train_losses.append(epoch_loss)
                        train_acc.append(epoch_acc.cpu())
                    else:
                        val_losses.append(epoch_loss)
                        val_acc.append(epoch_acc.cpu())

                    # deep copy the model,
                    if phase == 'val':
                        self.early_stopping(val_loss=epoch_loss)
                        torch.save(self.model.state_dict(), best_model_params_path)

                if self.early_stop == True:
                    print("Early Stopping")
                    break

                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Lowest Loss: {self.lowest_loss:4f}')

            # load best model weights
            self.model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
        
        self.plot_loss_curves(train_losses, val_losses, train_acc, val_acc)
        # return train_losses, val_losses

    def test_accuracy(self):
        self.model.eval()

        y_true = torch.tensor([])
        y_score = torch.tensor([])

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                labels = labels.squeeze()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                labels = labels.float().resize_(len(labels), 1)

                y_true = torch.cat((y_true, labels.cpu()), 0)
                y_score = torch.cat((y_score, preds.cpu()), 0)
        
            y_true = y_true.numpy()
            y_score = y_score.detach().numpy()

            evaluator = Evaluator("breastmnist", "test")
            metrics = evaluator.evaluate(y_score)

            print(f"Test auc: {metrics[0]:.3f}  acc: {metrics[1]:.3f}")

            cm = confusion_matrix(y_true, y_score)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.show()

    def visualize_model(self, num_images=6):
        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {self.class_names[preds[j].item()]}\n actual: {self.class_names[labels[j].item()]}')
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        # plt.show(block=False)
                        self.model.train(mode=was_training)
                        return
            self.model.train(mode=was_training)

    def save_model(self, model_filename="cnn_model_weights.pt"):

        if not model_filename.endswith(".pt"):
            print(f"Incorrect or missing file extension. Changing filename to '{model_filename}.pt'")
            # removes any other file extension and adds .pt
            model_filename = f"{os.path.splitext(model_filename)[0]}.pt" 

        model_path = os.path.join(os.getcwd(),"A", model_filename)
        torch.save(self.model, model_path)
        print(f"The model have been saved to {model_filename}")


def load_test():

    action = int(input("1. train or 2. load\nEnter 1 or 2\n").strip())

    # train options
    if action == 1:
        load_model = False
        model_choice = int(input("\nChoose a model:\n1. Resnet-18\n2. MobileNetV3\n3. EfficientNetV2\nEnter 1, 2 or 3\n").strip())
        optimiser_choice = int(input("\nChoose an optimiser\n1. SGD with Momentum\n2. Adam\n3. RMSProp\nEnter 1, 2 or 3\n").strip())
        scheduler_choice = int(input("\nChoose an scheduler\n1. StepLR\n2. None\nEnter 1 or 2\n").strip())
        learning_rate = float(input("\nEnter the learning rate (e.g. 0.001): \n").strip())
        epochs = int(input("Enter number of epochs: \n").strip())
        weighted_choice = int(input("Weighting off classes:\n1. no weights\n2. Balanced Weights\n").strip())

        if weighted_choice == 1:
            weighted = False
        else:
            weighted = True
        
    elif action == 2:
        load_model = True
        model_choice = 0
    
    aa = modelACNN(BATCH_SIZE=16, load_model=load_model, model_type=model_choice, patience=10, delta=0.01)

    if action == 1:
        optimizers = [
            optim.SGD,  # Stochastic Gradient Descent, with and without Momentum
            optim.Adam,  # Adam (Adaptive Moment Estimation)
            optim.RMSprop,  # Root Mean Square Propagation
        ]

        momentum = 0.9

        # Observe that all parameters are being optimized
        if optimiser_choice == 1:
            optimizer_ft = optimizers[optimiser_choice-1](aa.model.parameters(), lr=learning_rate, momentum = momentum)
        else:
            optimizer_ft = optimizers[optimiser_choice-1](aa.model.parameters(), lr=learning_rate)


        # Decay LR by a factor of 0.1 every 5 epochs
        if scheduler_choice == 1:
            scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
        else:
            scheduler = None

        aa.train_model(optimizer=optimizer_ft, scheduler=scheduler,num_epochs=epochs, weighted= weighted)
    
    aa.visualize_model()
    aa.test_accuracy()

    if action == 1:
        save_decision = input("Do you want to save the model? (Y/n): \n").strip().lower()

        if save_decision == 'y':
            # Prompt the user for the file name
            file_name = input("Enter the file name to save the model: ").strip()
            
            aa.save_model(file_name)
        elif save_decision == 'n':
            print("Model not saved.")
        else:
            print("Invalid input. Please enter 'Y' or 'n'.")

        plt.show()


if __name__ == "__main__":
    load_test()