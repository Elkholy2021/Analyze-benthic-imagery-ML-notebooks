from termcolor import cprint
from PIL import Image
from IPython.display import display
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt 
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random
from collections import Counter
import warnings
import os
import json
import shutil

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
class FineTuning():
    def __init__(self,
                 config,
                 config_finetuning,
                 seed = 42,
                 scheduler_type  ="Cosine Annealing Warm Restarts",
                suffix = '_finetuning',
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        self.config = config
        dataset_dir = config.annotations['download_dir']
        dataset_name = config.annotations['dataset_name']
        which_set = config.annotations['which_set']
        self.checkpoint_path = f"checkpoints/model_{dataset_name}.pt"
        self.dataset_name = dataset_name
        self.which_set = which_set
        self.dataset_dir = os.path.join(dataset_dir,dataset_name,which_set+suffix) 
        self.train_ratio = config_finetuning['train_ratio']
        self.valid_ratio = config_finetuning['valid_ratio']
        self.dropout_ratio = config_finetuning['dropout_ratio']
        self.freeze_layers = config_finetuning['freeze_layers']
        self.early_stopping = config_finetuning['early_stopping']
        self.learning_rate = config_finetuning['learning_rate']
        self.epochs = config_finetuning['epochs']
        model_arch = config_finetuning['model_arch']
        criterion_type = config_finetuning['criterion_type']
        optimizer_type = config_finetuning['optimizer_type']
        self.model_arch = model_arch
        self.criterion_type = criterion_type
        self.optimizer_type = optimizer_type
        self.dataset_name = dataset_name
        self.which_set = which_set
        self.basic_transform = transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor()
                                                ])
        
        self.available_models = {"resnet18":models.resnet18,
                                "resnet34":models.resnet34,
                                "resnet50":models.resnet50,
                                "inception_v3": models.inception_v3,
                                "efficientnet_b0": models.efficientnet_b0
                    }
        
        self.available_criterions = {"Cross Entropy":torch.nn.CrossEntropyLoss,
                                     "Negative Log Likelihood Loss": torch.nn.NLLLoss,
                                     "Binary Cross Entropy": torch.nn.BCELoss
                                    }
        self.available_optimizers = {"Adam": optim.Adam,
                                     "SGD": optim.SGD,
                                     "AdamW": optim.AdamW
                                     }
        

        self.available_schedulers = {"Cosine Annealing Warm Restarts":optim.lr_scheduler.CosineAnnealingWarmRestarts}


        self.model = self.available_models[model_arch]
        self.criterion = self.available_criterions[criterion_type]
        self.optimizer = self.available_optimizers[optimizer_type]
        self.scheduler = self.available_schedulers[scheduler_type]
        self.device = device

    def preprocess_dataset(self, dataset_dir=None, stats_save_path=None):
        """
        Compute mean and std for the dataset, save them to a JSON file,
        and define transform functions for training and evaluation.
        """
        if dataset_dir is None:
            dataset_dir = self.dataset_dir

        self.mean, self.std = self.compute_mean_std(dataset_dir)

        print("Dataset Mean:", self.mean)
        print("Dataset Std:", self.std)

        # Save mean and std to JSON
        stats = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist()
        }
        if stats_save_path == None:
            stats_save_path = os.path.join(os.path.dirname(self.checkpoint_path), os.path.basename(self.checkpoint_path).split('.')[0] + '.json')
            

        stats_path = Path(stats_save_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directories exist

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)

        # Define transformations
        # self.train_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        #     transforms.RandomRotation(10),
        #     transforms.GaussianBlur(kernel_size=3),
        #     transforms.ToTensor(),
        #     transforms.Normalize(self.mean.tolist(), self.std.tolist())
        # ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # simulate upside-down orientation
            transforms.RandomRotation(15),    # small-angle tilts
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # zoom and slight shift
            transforms.ToTensor(),
            transforms.Normalize(self.mean.tolist(), self.std.tolist())
        ])


        self.eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.mean.tolist(), self.std.tolist())
        ])

    def compute_mean_std(self,dataset_path,transform = None, batch_size=32, sample_limit=None):
        if transform == None:
            transform = self.basic_transform
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
    def load_dataset(self, 
                     dataset_dir = None,
                     train_ratio = None, 
                     valid_ratio = None,
                     train_transform = None,
                     eval_transform = None,
                     batch_size = 32,
                     seed = 42,
                     handling_data_imbalance = True

                     ):

        if dataset_dir == None:
            dataset_dir = self.dataset_dir
        if train_ratio == None:
            train_ratio = self.train_ratio
        if valid_ratio == None:
            valid_ratio = self.valid_ratio
        if train_transform == None:
            train_transform = self.train_transform
        if eval_transform == None:
            eval_transform = self.eval_transform

        full_dataset = datasets.ImageFolder(dataset_dir, transform=None)
        self.full_dataset = full_dataset
        total_size = len(self.full_dataset)
        train_size = int(train_ratio * total_size)
        valid_size = int(valid_ratio * total_size)
        test_size = total_size - train_size - valid_size
        num_classes = len(self.full_dataset.classes)

        train_dataset, valid_dataset, test_dataset = random_split(
            full_dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(seed))

        train_dataset.dataset.transform = train_transform
        valid_dataset.dataset.transform = eval_transform
        test_dataset.dataset.transform = eval_transform
        if handling_data_imbalance:
            self.class_weights = self.compute_class_weights_from_subset(train_dataset, num_classes)
            # print(f"class_weights: {self.class_weights}")
        else:
            self.class_weights = torch.ones(num_classes, dtype=torch.float)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    def freeze_backbone_and_modify_classifier(self,model, num_classes):
        """
        Freezes all layers of the model except for the first, penultimate, and last layers,
        and modifies the final classification layer to match the specified number of classes.

        Args:
            model (nn.Module): The pre-trained model to modify.
            num_classes (int): The number of output classes for the classification task.

        Returns:
            nn.Module: The modified model ready for fine-tuning.
        """
        # Get the immediate child modules of the model
        children = list(model.named_children())
        
        if len(children) < 4:
            print("Model has fewer than 3 layers; cannot freeze all but first, penultimate, and last.")
            return model

        # Freeze all parameters in the model
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze parameters in the first child module
        first_child_name, first_child = children[0]
        for param in first_child.parameters():
            param.requires_grad = True

        # Unfreeze parameters in the penultimate child module
        penultimate_child_name, penultimate_child = children[-2]
        for param in penultimate_child.parameters():
            param.requires_grad = True

        # Unfreeze parameters in the last child module
        last_child_name, last_child = children[-1]
        for param in last_child.parameters():
            param.requires_grad = True

        # Modify the final classification layer
        if isinstance(last_child, nn.Linear):
            in_features = last_child.in_features
            setattr(model, last_child_name, nn.Linear(in_features, num_classes))
        elif isinstance(last_child, nn.Sequential):
            # If the last child is a Sequential, replace the last Linear layer
            modules = list(last_child.children())
            for i in reversed(range(len(modules))):
                if isinstance(modules[i], nn.Linear):
                    in_features = modules[i].in_features
                    modules[i] = nn.Linear(in_features, num_classes)
                    break
            new_seq = nn.Sequential(*modules)
            setattr(model, last_child_name, new_seq)
        else:
            print("The last child module is not a Linear layer or Sequential containing a Linear layer.")
            return model

        print(f"All layers frozen except '{first_child_name}', '{penultimate_child_name}', and '{last_child_name}'.")
        print(f"Modified the final classification layer to have {num_classes} output classes.")

        return model
    def apply_dropout_to_resnet(self,model, dropout_p=0.3):
        """
        Injects Dropout into ResNet's layer4 after each residual block's ReLU.
        """
        for name, module in model.layer4.named_children():
            if hasattr(module, 'relu') and isinstance(module.relu, nn.ReLU):
                module.relu = nn.Sequential(
                    module.relu,
                    nn.Dropout(p=dropout_p)
                )
        return model
    def fine_tune_model(self,
                        model = None,
                        criterion = None,
                        optimizer  =None,
                        scheduler = None,
                        freeze_layers = None,
                        train_loader = None,
                        valid_loader = None,
                        test_loader = None,
                        epochs = None,
                        learning_rate = None,
                        checkpoint_path = None,
                        early_stopping = None,
                        device = None
                        ):
        if device == None:
            device = self.device
        if early_stopping == None:
            early_stopping = self.early_stopping
        if freeze_layers == None:
            freeze_layers = self.freeze_layers
        if epochs == None:
            epochs = self.epochs
        if learning_rate == None:
            learning_rate = self.learning_rate
        if model == None:
            model = self.model(pretrained = True)
            model = self.apply_dropout_to_resnet(model, dropout_p=self.dropout_ratio)  # Add dropout to backbone
        if criterion == None:
            criterion = self.criterion(weight=self.class_weights.to(device))
        if optimizer == None:
            optimizer = self.optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        if scheduler == None:
            scheduler = self.scheduler(optimizer, T_0=5)
        if train_loader == None:
            train_loader = self.train_loader
        if valid_loader == None:
            valid_loader = self.valid_loader
        if test_loader == None:
            test_loader = self.test_loader
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path

        num_classes = len(self.full_dataset.classes)

        # model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_ratio),  # Dropout before final layer
            nn.Linear(model.fc.in_features, num_classes)
        )

        if freeze_layers:
            # model = self.freeze_backbone_and_modify_classifier(model,num_classes)
            for param in model.parameters():
                param.requires_grad = False
            for param in model.conv1.parameters():
                param.requires_grad = True
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True



        model = model.to(device)
        print(f"{self.model_arch} Model loaded with pretrained weights from torchvision to {device}")
        # criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(device))
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)

        best_accuracy = 0.0
        patience = 3
        epochs_no_improve = 0
        train_acc_list, valid_acc_list = [], []

        for epoch in range(epochs):
            model.train()
            train_loss, correct_train, total_train = 0.0, 0, 0

            for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            train_accuracy = correct_train / total_train * 100
            train_acc_list.append(train_accuracy)

            model.eval()
            valid_loss, correct_valid, total_valid = 0.0, 0, 0

            with torch.no_grad():
                for images, labels in tqdm(valid_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    valid_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct_valid += (preds == labels).sum().item()
                    total_valid += labels.size(0)

            valid_accuracy = correct_valid / total_valid * 100
            valid_acc_list.append(valid_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%")

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved to {checkpoint_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience and early_stopping:
                    print("Early stopping triggered.")
                    break

            scheduler.step(epoch + 1)
        self.model = model
        print("Fine-tuning complete!\n")


    def evaluation(self,
                   model = None,
                   checkpoint_path = None,
                   test_loader = None,
                   device = None
                   ):
        if model == None:
            model = self.model
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path
        if test_loader == None:
            test_loader = self.test_loader
        if device == None:
            device = self.device
        print("\nEvaluating on test set...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            model.load_state_dict(torch.load(checkpoint_path))

        model.eval()

        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        class_names = self.full_dataset.classes
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix on Test Set')
        plt.tight_layout()
        plt.show()
    def predict_image(self, image_path):

        try:
            img = Image.open(image_path).convert('RGB')
            input_tensor = self.eval_transform(img).unsqueeze(0).to(self.device)
        except:
            print(f"There is a problem with this image {image_path}")
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()


    def classify_images0(self, image_dir = None, output_dir = None, class_labels=None):
        if image_dir == None:
            image_dir = os.path.join(self.config.annotations['download_dir'],self.config.annotations['dataset_name'],'inference_dataset')
        if output_dir == None:
            output_dir = os.path.join(self.config.annotations['download_dir'],self.config.annotations['dataset_name'],'classification_inference_results')
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_path in tqdm(image_dir.glob('*.*'),desc = 'Inferring...'):
            label_idx = self.predict_image(image_path)
            label_name = class_labels[label_idx] if class_labels else str(label_idx)

            dest_folder = output_dir / label_name
            dest_folder.mkdir(parents=True, exist_ok=True)

            shutil.copy(image_path, dest_folder)


    def classify_images(self, whole_frame_annotations,point_annotations,image_dir=None, output_dir=None, class_labels=None):
        if class_labels == None:
            if self.config.annotations['which_set'] == "whole_frame_annotations":
                class_labels = [item.replace('/', '') for item in whole_frame_annotations]
            elif self.config.annotations['which_set'] == "point_annotations":
                class_labels = [item.replace('/', '') for item in point_annotations]
        if image_dir is None:
            image_dir = os.path.join(
                self.config.annotations['download_dir'],
                self.config.annotations['dataset_name'],
                self.config.annotations['which_set']+'_inference'
            )
        if output_dir is None:
            output_dir = os.path.join(
                self.config.annotations['download_dir'],
                self.config.annotations['dataset_name'],
                'results_'+self.config.annotations['which_set']+'_inference'
            )

        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define valid image extensions
        valid_exts = ('.png', '.jpg', '.jpeg')

        # Recursively search all image files
        all_images = [p for p in image_dir.rglob('*') if p.suffix.lower() in valid_exts]

        for image_path in tqdm(all_images, desc='Inferring...'):
            try:
                label_idx = self.predict_image(image_path)
            except:
                print(image_path)
            label_name = class_labels[label_idx] if class_labels else str(label_idx)

            dest_folder = output_dir / label_name
            dest_folder.mkdir(parents=True, exist_ok=True)

            shutil.copy(image_path, dest_folder)