import os
from termcolor import cprint
from PIL import Image
from IPython.display import display
import requests
import gdown
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from collections import Counter
class utils():
    def __init__(self,config):
        self.api_token_path = config.api_token['path']
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        self.config = config
        
        # self.ML_model_path_debris = config.models['ADM_path']
        # self.ML_model_path_debris_TR30 = config.models['ADM_path_TR30']
        # self.ML_model_url_debris = config.models['ADM_url']
        # self.ML_model_url_debris_TR30 = config.models['ADM_url_TR30']

        # self.ML_model_path_sam = config.models['sam_model_path']
        # self.ML_model_url_sam = config.models['sam_model_url']
        # self.models_dir = config.models['directoty']
    def check_directory_exists(self,path):
        """
        Ensure that a directory exists. Create it if it doesn't.

        Parameters:
        - path: str or Path — directory path to check/create
        """
        dir_path = Path(path).expanduser().resolve()
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def get_directory_from_filepath(self,filepath):
        """
        Extract the directory from a full file path.

        Parameters:
        - filepath: str or Path — full path to a file

        Returns:
        - Path object of the directory containing the file
        """
        return Path(filepath).expanduser().resolve().parent
    def check_api_token(self,api_token_path = None):
        if api_token_path == None:
            api_token_path = self.api_token_path

        if os.path.isfile(api_token_path) and os.path.getsize(api_token_path) > 0:
            cprint("API token Found ✅", "green")
        else:
            cprint("API token not found", "red")
            print("Please enter your api token:")
            self.show_instruction_figure('figures/how_to_get_api_token.png')
            api_token = input("> ")
            file_path = self.api_token_path

            self.check_directory_exists(self.get_directory_from_filepath(file_path))

            with open(file_path, "w") as f:
                f.write(api_token)

            cprint(f"API token saved to '{file_path}' ✅", "green")
    def check_ML_model(self,ML_model_path_debris = None,ML_model_path_debris_TR30 = None,ML_model_path_sam = None):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            print(f"Directory created: {self.models_dir}")
        else:
            print(f"Directory already exists: {self.models_dir}")
            
        if ML_model_path_debris is None:
            ML_model_path_debris = self.ML_model_path_debris
        if os.path.isfile(ML_model_path_debris):
            cprint("Debris model TR80 Found ✅", "green")
        else:
            cprint("Debris model TR80 not found", "red")
            print("Downloading the debris model TR80")
            file_id = self.ML_model_url_debris.split("/d/")[1].split("/")[0]
            gdown_url = f"https://drive.google.com/uc?id={file_id}"

            gdown.download(gdown_url, ML_model_path_debris, quiet=False)

        if ML_model_path_debris_TR30 is None:
            ML_model_path_debris_TR30 = self.ML_model_path_debris_TR30
        if os.path.isfile(ML_model_path_debris_TR30):
            cprint("Debris model TR30 Found ✅", "green")
        else:
            cprint("Debris model TR30 not found", "red")
            print("Downloading the debris model TR30")
            file_id = self.ML_model_url_debris_TR30.split("/d/")[1].split("/")[0]
            gdown_url = f"https://drive.google.com/uc?id={file_id}"

            gdown.download(gdown_url, ML_model_path_debris_TR30, quiet=False)

        if ML_model_path_sam is None:
            ML_model_path_sam = self.ML_model_path_sam
        if os.path.isfile(ML_model_path_sam):
            cprint("SAM model Found ✅", "green")
        else:
            cprint("SAM model not found", "red")
            print("Downloading the SAM model")
            file_id = self.ML_model_url_sam.split("/d/")[1].split("/")[0]
            gdown_url = f"https://drive.google.com/uc?id={file_id}"

            gdown.download(gdown_url, ML_model_path_sam, quiet=False)
    def show_instruction_figure(self,figure_path):


        image = Image.open(figure_path)  # Replace with your image path

        # Display it in the notebook
        display(image)

    def compute_mean_std(self,dataset_path,transform = None, batch_size=32, sample_limit=None):
        if transform == None:
            transform = self.transform

        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        mean = 0.
        std = 0.
        nb_samples = 0.

        for i, (data, _) in enumerate(tqdm(dataloader, desc="Computing mean and std")):
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)  # [B, C, H*W]
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

            if sample_limit and nb_samples >= sample_limit:
                break

        mean /= nb_samples
        std /= nb_samples

        return mean, std
    
    def compute_class_weights_from_subset(self,train_subset, num_classes):
        """
        Compute class weights from a torch.utils.data.Subset (e.g. train_dataset).
        
        Args:
            train_subset (torch.utils.data.Subset): The subset containing training data.
            num_classes (int): Total number of classes in the dataset.

        Returns:
            torch.Tensor: Tensor of class weights.
        """
        # Get the class indices (targets) from the original dataset
        targets = [train_subset.dataset.targets[i] for i in train_subset.indices]
        
        # Compute weights using sklearn
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(num_classes),
            y=targets
        )

        return torch.tensor(class_weights, dtype=torch.float)
    
    def compute_dataset_statistics(self,
                                   dataset_dir = None,
                                   dataset_name = None,
                                   which_set = None,
                                   suffix = '_finetuning'
                                   ):
        """
        Computes basic statistics of an image dataset stored in `dataset_dir`.

        Args:
            dataset_dir (str): Path to the dataset root (expected to be ImageFolder-style).
            sample_size (int): Number of random samples to estimate mean and std.

        Returns:
            dict: A dictionary containing class distribution, mean, std, total_size, and num_classes.
        """
        if dataset_dir == None:
            dataset_dir = self.config.annotations['download_dir']

        if dataset_name == None:
            dataset_name = self.config.annotations['dataset_name']

        if which_set == None:
            which_set = self.config.annotations['which_set']

        dataset_dir =  os.path.join(dataset_dir,dataset_name)
        dataset_dir =  os.path.join(dataset_dir,which_set+suffix)

        dataset = datasets.ImageFolder(dataset_dir, transform=None)

        total_size = len(dataset)
        num_classes = len(dataset.classes)

        # Class distribution
        class_counts = Counter([label for _, label in dataset.samples])
        class_distribution = {dataset.classes[i]: count for i, count in class_counts.items()}

        # Mean and std estimation (sample a subset for performance)
        sample_size = total_size
        indices = torch.randperm(total_size)[:sample_size]


        for idx in tqdm(indices):
            try:
                img_path, _ = dataset.samples[idx]
                img = Image.open(img_path).convert('RGB')
                img_tensor = transforms.ToTensor()(img)
            except:
                print(f"There is a problem with this image: {img_path} ")

        # stats = {
        #     "total_images": total_size,
        #     "num_classes": num_classes,
        #     "class_distribution": class_distribution,
        # }

        # Print summary
        print("📊 Dataset Statistics:")
        print(f"  - Total Images: {total_size}")
        print(f"  - Number of Classes: {num_classes}")
        print(f"  - Class Distribution: {class_distribution}")




class FineTuningParameters:
    def __init__(self, on_config_applied=None):
        # Initialize widgets
        self.freeze_layers_widget = widgets.Checkbox(value=True, description="Freeze Pretrained Layers")
        self.early_stopping_widget = widgets.Checkbox(value=True, description="Use Early Stopping")
        self.epochs_widget = widgets.IntSlider(value=50, min=1, max=100, step=1, description="Epochs")
        self.batch_size_widget = widgets.IntSlider(value=32, min=8, max=128, step=8, description="Batch Size")
        self.learning_rate_widget = widgets.FloatLogSlider(value=1e-4, base=10, min=-5, max=-2, step=0.1, description="Learning Rate")
        self.train_ratio_widget = widgets.FloatSlider(value=0.8, min=0.05, max=0.9, step=0.05, description="Train Ratio")
        self.valid_ratio_widget = widgets.FloatSlider(value=0.1, min=0.05, max=0.3, step=0.05, description="Valid Ratio")
        self.dropout_ratio_widget = widgets.FloatSlider(value=0.5, min=0.00, max=0.7, step=0.05, description="Dropout ratio")

        self.model_arch_widget = widgets.Dropdown(
            options=["resnet18", "resnet34", "resnet50"],
            value="resnet18",
            description="Model Architecture"
        )
        self.criterion_type_widget = widgets.Dropdown(
            options=["Cross Entropy", "Negative Log Likelihood"],
            value="Cross Entropy",
            description="Loss Function"
        )
        self.optimizer_type_widget = widgets.Dropdown(
            options=["Adam", "SGD", "AdamW"],
            value="Adam",
            description="Optimizer"
        )
        self.output = widgets.Output()
        self.apply_button = widgets.Button(description="Apply", button_style='info')
        self.apply_button.on_click(self.on_apply_clicked)
        self.config = None  # To store the configuration after applying
        self.on_config_applied = on_config_applied  # Callback function

    def display_widgets(self):
        # Arrange widgets in a layout
        param_widgets = widgets.VBox([
            widgets.HTML("<h3>🔧 Model Training Configuration</h3>"),
            widgets.HBox([self.freeze_layers_widget, self.early_stopping_widget]),
            self.epochs_widget,
            self.batch_size_widget,
            self.learning_rate_widget,
            widgets.HBox([self.train_ratio_widget, self.valid_ratio_widget, self.dropout_ratio_widget]),
            self.model_arch_widget,
            self.criterion_type_widget,
            self.optimizer_type_widget,
            self.apply_button,
            self.output
        ])
        display(param_widgets)

    def on_apply_clicked(self, b):
        with self.output:
            clear_output()
            # Extract widget values on click
            self.config = {
                "freeze_layers": self.freeze_layers_widget.value,
                "early_stopping": self.early_stopping_widget.value,
                "epochs": self.epochs_widget.value,
                "batch_size": self.batch_size_widget.value,
                "learning_rate": self.learning_rate_widget.value,
                "train_ratio": self.train_ratio_widget.value,
                "valid_ratio": self.valid_ratio_widget.value,
                "model_arch": self.model_arch_widget.value,
                "criterion_type": self.criterion_type_widget.value,
                "optimizer_type": self.optimizer_type_widget.value,
                "dropout_ratio": self.dropout_ratio_widget.value
            }
            print("🟢 Configuration accepted:")
            for k, v in self.config.items():
                print(f"{k}: {v}")
            if self.on_config_applied:
                self.on_config_applied(self.config)

    def get_config(self):
        return self.config