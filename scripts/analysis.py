import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from matplotlib.patches import Rectangle
import folium
import contextily as ctx
import math
import ipywidgets as widgets
from IPython.display import display, clear_output
import random
import cv2
import torch
import warnings
from segment_anything import sam_model_registry, SamPredictor

class analysis():

    def __init__(self,config):
        self.data = None
        self.csv_file_path = config.annotations['annotations_dir']
        self.m = None
        self.config = config
        self.label_colors = None
    def load_csv(self,csv_file_path = None):
        if csv_file_path == None:
            csv_file_path = self.csv_file_path
        try:
            data = pd.read_csv(csv_file_path)
            self.data = data
        except FileNotFoundError:
            print(f"Error: The file '{csv_file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
    def stratified_split(self,df, label_col='label.name', sample_fraction= None, random_state=42):
        """
        Splits the DataFrame into a stratified N% sample per label and the remaining 1-N%.

        Args:
            df (pd.DataFrame): Original DataFrame.
            label_col (str): Column name containing the labels.
            sample_fraction (float): Fraction to sample per label (default 0.1).
            random_state (int, optional): For reproducibility.

        Returns:
            df_sampled (pd.DataFrame): Stratified sampled N%.
            df_remaining (pd.DataFrame): Remaining 1-N%.
        """
        if sample_fraction == None:
            sample_fraction = float(self.config.annotations['split_set_ratio'])
        # Get N% of each label group
        df_sampled = df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=random_state)
        )

        # Get the remaining 100-N%
        df_remaining = df.drop(index=df_sampled.index)
        print(f"Sampled: {len(df_sampled)} rows, Remaining: {len(df_remaining)} rows")

        return df_sampled, df_remaining



    def show_sample_images(self, labels = None,df=None):
        if df is None:
            df = self.data

        if labels is None:
            labels = ["Anthropogenic Debris", "Natural"]

        # Filter rows with valid image URLs
        df_valid = df.dropna(subset=["point.media.path_best"])
        df_valid = df_valid[df_valid["point.media.path_best"].str.startswith("http")]

        # Pick 6 random images
        sample = df_valid.sample(n=6)

        # Generate random colors for each label
        available_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                            'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                            'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
                            'gray', 'black', 'lightgray']
        if self.label_colors == None:
            label_colors = {label: random.choice(available_colors) for label in labels}
            self.label_colors = label_colors
        else:
            label_colors = self.label_colors
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for i, (idx, row) in enumerate(sample.iterrows()):
            url = row["point.media.path_best"]
            raw_label = str(row.get("label.name", ""))

            # Assign label from the input list
            assigned_label = next((label for label in labels if label in raw_label), "Other")
            border_color = label_colors.get(assigned_label, "gray")

            try:
                response = requests.get(url,timeout=10)
                img = Image.open(BytesIO(response.content))

                axes[i].imshow(img)
                axes[i].set_title(f"Label: {assigned_label}", fontsize=9)
                axes[i].axis('off')

                # Draw a rectangle border manually
                axes[i].add_patch(Rectangle(
                    (0, 0), 1, 1,
                    transform=axes[i].transAxes,
                    fill=False,
                    edgecolor=border_color,
                    linewidth=6
                ))

            except Exception as e:
                axes[i].set_title("Image load error", fontsize=9)
                axes[i].axis('off')

        plt.tight_layout()
        plt.show()


    def build_folium_map(self, labels = None,df=None):
        if df is None:
            df = self.data



        df = df.dropna(subset=["point.pose.lat", "point.pose.lon", "point.media.path_best"])

        # Center the map
        center_lat = df["point.pose.lat"].mean()
        center_lon = df["point.pose.lon"].mean()

        # Create the folium map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles="OpenStreetMap",
            max_zoom=25
        )

        # Generate random colors for each label
        available_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                            'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                            'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
                            'gray', 'black', 'lightgray']

        label_colors = {label: random.choice(available_colors) for label in labels}

        # Add markers with images and dynamically assigned labels and colors
        for _, row in df.iterrows():
            raw_label = str(row.get("label.name", ""))
            img_url = row.get("point.media.path_best", "")

            assigned_label = next((label for label in labels if label in raw_label), "Other")
            color = label_colors.get(assigned_label, "gray")

            popup_html = f"""
            <strong>Label:</strong> {assigned_label}<br>
            <img src="{img_url}" width="300">
            """

            folium.Marker(
                location=[row["point.pose.lat"], row["point.pose.lon"]],
                popup=folium.Popup(popup_html, max_width=400),
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(m)

        # Assign map to self
        self.m = m

    def show_folium_map(self, m = None):
        if m is None:
            m = self.m
        return m
    def show_ctx_providers(self):
        print("All providers:")
        print(list(ctx.providers.keys()))
        print("Providers in OpenStreetMap:")
        print(list(ctx.providers['OpenStreetMap'].keys()))
        print("Providers in Esri:")
        print(list(ctx.providers['Esri'].keys()))
    def compute_average_altitude(self,df):

        if 'point.pose.alt' not in df.columns:
            print("Column 'point.pose.alt' not found in the DataFrame.")
            return None

        mean_alt = df['point.pose.alt'].dropna().mean()

        return mean_alt

    def get_dict_with_xy_data(self,df):

        dict = {}

        for _, row in df.iterrows():
            x = row.get("point.x")
            y = row.get("point.y")
            alt = row.get("point.pose.alt")
            dep = row.get("point.pose.dep")
            lat = row.get("point.pose.lat")
            lon = row.get("point.pose.lon")

            path = row.get("point.media.path_best")

            if pd.notnull(x) and pd.notnull(y) and pd.notnull(path):
                dict[path] = [x, y,alt,dep,lat,lon]

        return dict

    def get_dict_with_all_data(self,df):

        dict = {}

        for _, row in df.iterrows():
            x = row.get("point.x")
            y = row.get("point.y")
            alt = row.get("point.pose.alt")
            dep = row.get("point.pose.dep")
            lat = row.get("point.pose.lat")
            lon = row.get("point.pose.lon")

            path = row.get("point.media.path_best")

            dict[path] = [x, y,alt,dep,lat,lon]

        return dict


def compute_area_cm(
        altitude,
        pixel_area,
        config
        ):
    """ 
    HFOV and VFOV were calculated previously for the external blueye camera while underwater 
    based on the specs published on their website
    
    image_width, image_height are fixed for all images taken by this camera

    The tilt angle was part of the mechanical design, camera was at the front side pointing downward
    with 15deg inclination inwards.
    """
    HFOV = float(config.camera_params['HFOV'])
    VFOV = float(config.camera_params['VFOV'])
    image_width = int(config.camera_params['image_width'])
    image_height = int(config.camera_params['image_height'])
    tilt_deg = float(config.camera_params['tilt_deg'])
    # Convert degrees to radians
    hfov_rad = math.radians(HFOV)
    vfov_rad = math.radians(VFOV)
    tilt_rad = math.radians(tilt_deg)

    half_vfov = vfov_rad / 2
    half_hfov = hfov_rad / 2

    # Slant range to top and bottom edge of image
    range_top = altitude / math.cos(tilt_rad - half_vfov)
    range_bottom = altitude / math.cos(tilt_rad + half_vfov)

    # Ground distance
    y_top = range_top * math.sin(tilt_rad - half_vfov)
    y_bottom = range_bottom * math.sin(tilt_rad + half_vfov)

    
    footprint_height = abs(y_bottom - y_top)
    footprint_width = 2 * altitude * math.tan(half_hfov)

    # Resolution in cm/pixel
    res_x_cm = (footprint_width / image_width) * 100
    res_y_cm = (footprint_height / image_height) * 100

    return res_x_cm * res_y_cm * pixel_area





