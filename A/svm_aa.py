import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from skimage.feature import hog
import time

from torchvision import transforms
import joblib

import medmnist
from medmnist import INFO, BreastMNIST

import os

class modelASVM():
    def __init__(self, load = False, hog = False):
        # have two classifiers, one loaded from the directory, and one that can be used to train
        self.clf = None
        self.hog = hog

        if not load:
            self.input, self.labels, self.dataset_sizes, self.class_names = self.load_dataset()
            self.clf, self.best_params = self.create_classifier()
        else:
            self.clf = self.load_model()
            self.input, self.labels, self.dataset_sizes, self.class_names = self.load_dataset()
    
    def process_image(self, image):
        if self.hog:
            fd = hog(
                image,
                orientations=6,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=False,
                channel_axis=None,
            )
            return fd
        else:
            return image.flatten()

    def load_dataset(self):
        """
        Function to load the data the BreastMNIST data set into Numpy Array
        Arg:
            hog: if true will return hog features, if false flattens the image
        Returns:
            images: dictionary containing images for all 3 data splits, images are flattened
            labels: dictionary containing labels for all 3 data splits
            dataset_sizes: dictionary of split sizes
            class_names: class map dictionary
        """
        print("Loading dataset...")

        # get the path for the datasets
        dir = os.getcwd()

        data_flag = "breastmnist"
        data_dir = "Datasets"

        PATH = os.path.join(dir, data_dir)

        data_transforms = {
            "train": transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Lambda(lambda x: x.squeeze().numpy()),
                transforms.Lambda(self.process_image)
            ]),
            "val": transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.squeeze().numpy()),
                transforms.Lambda(self.process_image)
            ]),
            "test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.squeeze().numpy()),
                transforms.Lambda(self.process_image)
            ]),
        }

        # retrieve the data from the 
        datasets = {x: BreastMNIST(split= x, transform=data_transforms[x], download=False, root=PATH)
                    for x in ["train", "val", "test"]} 
        
        # get the size of all 3 data splits
        dataset_sizes = {x: len(datasets[x]) for x in ["train", "val", "test"]}

        # process the images
        inputs = {x: np.array([datasets[x][i][0] for i in range(dataset_sizes[x])])
            for x in ["train", "val", "test"]}
        
        labels = {x: np.array([datasets[x][i][1] for i in range(dataset_sizes[x])]).ravel()
                    for x in ["train", "val", "test"]}

        print(f"Shape of inputs: {inputs["train"][0].shape}")

        # load the class names
        info = INFO[data_flag]
        class_names = info["label"]
        class_names = {int(key): value for key, value in class_names.items()}

        return inputs, labels, dataset_sizes, class_names
    
    def create_classifier(self):
        """
        Performs cross validation to pick the best hyperparameters and kernels for the dataset
        """
        print("Training starting...")
        since = time.time()
        base_clf = svm.SVC()

        p_grid = {"C": np.logspace(-2.5, 2, 10),
                  "degree": [3, 4, 5],
                  "gamma": [0.01, 0.1, "auto", "scale"],
                  "kernel": ["linear", "poly", "rbf"],
                  "class_weight": [None, "balanced", {0: 0.7, 1: 0.3}]
                  }

        # track the highest scores for each iteration
        scores = []

        #  track best overall parameters and the model
        highest_score = 0
        best_svm = None
        best_params = None
        iterations = 5

        for i in range (iterations):
            print(f"Iteration {i}/{iterations - 1}")
            print('-' * 10)
            
            # split into folds using stratified k fold
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

            # cross validation 
            clf = GridSearchCV(estimator=base_clf, param_grid=p_grid, cv=inner_cv)

            # fit the svm
            clf.fit(np.concatenate((self.input["train"], self.input["val"]), axis=0), 
                    np.concatenate((self.labels["train"], self.labels["val"]), axis=0))
            
            scores.append(clf.best_score_)

            print(f"Iteration {i}, best score: {clf.best_score_}\n")

            # update the values if the 
            if clf.best_score_ > highest_score:
                highest_score = clf.best_score_
                best_params = clf.best_params_
                best_svm = clf.best_estimator_

        # track the total time of training
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        print(f"C: {best_params["C"]}, Kernel: {best_params["kernel"]}")
        print(f"gamma: {best_params["gamma"]}, degree: {best_params["degree"]}")
        print(f"class weights: {best_params["class_weight"]}")

        self.plot_accuracy(scores)
        return best_svm, best_params
    
    def plot_accuracy(self, scores):
        plt.figure()
        plt.plot(range(1, len(scores) + 1), scores, marker='o', label="Scores")
        plt.title("Highest accuracy for each iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        # plt.ylim(0,1)
        plt.grid()
        # plt.show()
    
    def get_test_accuracy(self):
        acc = self.clf.score(self.input["test"], self.labels["test"])
        predictions = self.clf.predict(self.input["test"])
        cm = confusion_matrix(self.labels["test"], predictions)

        decision_scores = self.clf.decision_function(self.input["test"])
        auc = roc_auc_score(self.labels["test"], decision_scores)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")

        print(f"Test acc = {acc}, auc = {auc}")

    def save_model(self, filename = "svm_model_a.pkl"):
        # if file extension is wrong or no file extension
        if not filename.endswith(".pkl"):
            print(f"Incorrect or missing file extension. Changing filename to '{filename}.pkl'")
            # removes any other file extension and adds .pkl
            filename = f"{os.path.splitext(filename)[0]}.pkl" 
        
        filepath = os.path.join(os.getcwd(), "A", filename)
        if self.clf is None:
            raise ValueError("There is no model saved")
        
        data = {
            "model": self.clf,
            "hog": self.hog
        }

        joblib.dump(data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self):
        files = [file for file in os.listdir(os.path.join(os.getcwd(),"A")) if file.endswith(".pkl")]
        print("Enter the number file that you want")
        for i, file in enumerate(files):
            print(f"{i}: {file}")

        choice = int(input("Please enter a number\n").strip())
        filename = files[choice]

        filepath = os.path.join(os.getcwd(), "A", filename)
    
        data = joblib.load(filepath)
        model = data["model"]
        self.hog = data["hog"]

        print(f"Model is loaded from {filepath}")

        params = model.get_params()
        print("The loaded model parameters:")
        print(f"C: {params["C"]}, Kernel: {params["kernel"]}")
        print(f"gamma: {params["gamma"]}, degree: {params["degree"]}")
        print(f"class weights: {params["class_weight"]}")
        return model
    
def load_test():
    action = int(input("1. train or 2. load\nEnter 1 or 2\n").strip())

    if action == 1:
        feat = int(input("How to input data?\n1. Flatten\n2. HOG feature extraction\n").strip())
        if feat == 2:
            HOG = True
        else:
            HOG = False
        load = False
    elif action == 2:
        load = True
        HOG = False

    # initalise
    asvm = modelASVM(load=load, hog=HOG)
    asvm.get_test_accuracy()

    if action == 1:
        save_decision = input("Do you want to save the model? (Y/n): \n").strip().lower()

        if save_decision == 'y':
            # Prompt the user for the file name
            file_name = input("Enter the file name to save the model: ").strip()
            
            asvm.save_model(file_name)
        elif save_decision == 'n':
            print("Model not saved.")
        else:
            print("Invalid input. Please enter 'Y' or 'n'.")
    
    plt.show()    


if __name__ == "__main__":
    load_test()