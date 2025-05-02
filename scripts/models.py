
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import random
import warnings
from IPython.display import display, clear_output
import ipywidgets as widgets
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from scripts.analysis import compute_area_cm

class anthropogenic_debris_model():
    def __init__(self,config, model_type = "TR30"):
        if model_type == "TR80":
            self.model_path = config.models['ADM_path']
        elif model_type == "TR30":
            self.model_path = config.models['ADM_path_TR30']
        self.class_names = ["Anthropogenic Debris", "Natural"]
        self.model = None
        self.config = config
    def load_model(self,model_path = None):
        if model_path is None:
            model_path = self.model_path
        num_classes = len(self.class_names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore all warnings in this block

            model = models.resnet50(pretrained=False)  # pretrained warning will be suppressed
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))  # FutureWarning suppressed

        model.eval()
        self.model = model
        return model

    def preprocess_image(self,image_url):
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.config.ADM_transform['mean'], std = self.config.ADM_transform['std'])
        ])

        return transform(image).unsqueeze(0), image  # Return both tensor and raw image

    def predict_random_image(self,df):
        model = self.model
        class_names = self.class_names
        # Filter for valid image URLs
        df_valid = df.dropna(subset=["point.media.path_best"])
        df_valid = df_valid[df_valid["point.media.path_best"].str.startswith("http")]
        
        # Select a random row
        row = df_valid.sample(n=1).iloc[0]
        # row = df_valid.iloc[914]
        image_url = row["point.media.path_best"]
        true_label = str(row.get("label.lineage_names", "Unknown"))

        if "Anthropogenic Debris" in true_label:
            true_label = "Anthropogenic Debris"
        else:
            true_label = "Natural"
        print(f"🖼️ Using image: {image_url}")
        print(f"✅ True label: {true_label}")

        # Preprocess and predict
        input_tensor, raw_image = self.preprocess_image(image_url)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        predicted_label = class_names[pred_class] if class_names else str(pred_class)
        confidence = probs[0][pred_class].item()

        # Plot image with prediction title
        plt.figure(figsize=(6, 6))
        plt.imshow(raw_image)
        plt.axis('off')
        plt.title(f"True label: {true_label}\nPredicted: {predicted_label} ", fontsize=12)
        plt.show()

    def ADM_demo(self,df):
        button = widgets.Button(description="🔁 Try another image!", button_style='info')
        def on_button_click(_):
            clear_output(wait=True)
            display(button)
            self.predict_random_image(df)
        button.on_click(on_button_click)
        display(button)
        self.predict_random_image(df)

    def inferernce_image(self,image_url, true_label):
        model = self.model
        class_names = self.class_names
        if "Anthropogenic Debris" in true_label:
            true_label = "Anthropogenic Debris"
        else:
            true_label = "Natural"
        print(f"🖼️ Using image: {image_url}")
        print(f"✅ True label: {true_label}")

        # Preprocess and predict
        input_tensor, raw_image = self.preprocess_image(image_url)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        predicted_label = class_names[pred_class] if class_names else str(pred_class)
        confidence = probs[0][pred_class].item()

        # Plot image with prediction title
        plt.figure(figsize=(6, 6))
        plt.imshow(raw_image)
        plt.axis('off')
        plt.title(f"True label: {true_label}\nPredicted: {predicted_label}", fontsize=12)
        plt.show()

class segmentation_model():
    def __init__(self,config):
        self.sam_checkpoint = config.models['sam_model_path']
        self.config = config
    def interactive_segmentation_widgets(self,data_dict, segment_objects_fn):
        def launch():
            clear_output(wait=True)

            # Pick a random image
            image_url = random.choice(list(data_dict.keys()))
            print(f"🖼️ Image URL: {image_url}")

            # Load image
            response = requests.get(image_url)
            image_pil = Image.open(BytesIO(response.content)).convert("RGB")
            image_np = np.array(image_pil)
            h, w = image_np.shape[:2]

            # UI Elements
            x_slider = widgets.IntSlider(min=0, max=w-1, value=w // 2, description='X:')
            y_slider = widgets.IntSlider(min=0, max=h-1, value=h // 2, description='Y:')
            segment_button = widgets.Button(description="📐 Segment!", button_style='success')
            another_button = widgets.Button(description="🔁 Try another image!", button_style='info')
            output_plot = widgets.Output()
            output_text = widgets.Output()

            # Display function
            def update_display(x, y):
                with output_plot:
                    clear_output(wait=True)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(image_np)
                    ax.scatter(x, y, c='red', s=100, marker='x')
                    ax.set_title("Select a point using sliders", fontsize=14)
                    ax.axis("off")
                    plt.tight_layout()
                    plt.show()

            # Segment button callback
            def on_segment_clicked(b):
                px = x_slider.value
                py = y_slider.value
                norm_x = px / w
                norm_y = py / h
                with output_text:
                    clear_output(wait=True)
                    print(f"🔍 Segmenting at ({px}, {py}) → norm: ({norm_x:.4f}, {norm_y:.4f})")
                    area_pixel = segment_objects_fn(image_url, norm_x, norm_y, alt=data_dict[image_url][2])
                    print(f"📏 Area: {area_pixel} pixels")

            # Another image button callback
            def on_another_clicked(b):
                launch()

            # Connect buttons
            segment_button.on_click(on_segment_clicked)
            another_button.on_click(on_another_clicked)

            # Display layout
            button_row = widgets.HBox([segment_button, another_button])
            control_block = widgets.VBox([x_slider, y_slider, button_row])
            display(control_block, output_plot, output_text)
            update_display(x_slider.value, y_slider.value)
            widgets.interactive_output(update_display, {'x': x_slider, 'y': y_slider})

        launch()


    def segment_objects(self,
                        image_url, 
                        norm_x, 
                        norm_y,
                        compute_area_cm_fn = compute_area_cm,
                        alt = 1,
                        sam_checkpoint=None,
                        model_type="vit_h",
                        return_area=True):
        if sam_checkpoint is None:
            sam_checkpoint = self.sam_checkpoint
        alt_assumption = False
        alt_avergage = 0.89 #calculated it from the dataframe
        if np.isnan(alt):
            alt = alt_avergage
            alt_assumption = True #if valid 
        # Load image
        response = requests.get(image_url)
        image_pil = Image.open(BytesIO(response.content)).convert("RGB")
        image_rgb = np.array(image_pil)

        # Convert normalized point to pixel
        img_h, img_w = image_rgb.shape[:2]
        px = int(norm_x * img_w)
        py = int(norm_y * img_h)
        input_point = np.array([[px, py]])
        input_label = np.array([1])

        # Load SAM model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)        
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)    
        sam.to(device=device)
        predictor = SamPredictor(sam)

        # Predict
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        # Show and compute area

        for i, mask in enumerate(masks):
            area_pixels = np.sum(mask)
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            plt.imshow(mask, alpha=0.5)
            plt.scatter(px, py, c='red', s=100, marker='x')
            # plt.title(f"Mask {i+1} - Score: {scores[i]:.3f}, Area: {area_pixels} pixels, Altitude: {alt: .2f} m, Area {compute_area_cm_fn(alt, area_pixels):.2f} cm2")
            plt.title(f"Pixel area: {area_pixels} pixels, Altitude: {'~' if alt_assumption else ''}{alt: .2f} m, Area {'~'}{compute_area_cm_fn(alt, area_pixels, self.config):.2f} cm2")

            plt.axis('off')
            plt.show()

            if return_area:
                return area_pixels
