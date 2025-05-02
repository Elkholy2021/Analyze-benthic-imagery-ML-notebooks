# %%
import csv
import requests
import os
import time
from termcolor import cprint
from scripts.utils import utils
from sqapi.api import SQAPI
import pandas as pd
from tqdm import tqdm


from PIL import Image
class SQAPI_requests():
    def __init__(self,
                 config,
                 api_key = None,
                 host = "https://squidle.org/", 
                 include_columns=["label.id","label.uuid","label.name","label.lineage_names","comment","needs_review","tag_names","updated_at","point.id","point.x","point.y","point.t","point.is_targeted","point.media.id","point.media.key","point.media.path_best","point.pose.timestamp","point.pose.lat","point.pose.lon","point.pose.alt","point.pose.dep","point.media.deployment.key","point.media.deployment.campaign.key"],
                 template= 'data.csv',
                 f={"operations":[{"module":"pandas","method":"json_normalize"}]},
                 q={"filters":[{"name":"label_id","op":"is_not_null"}]},
                 translate={"vocab_registry_keys": ["worms", "caab", "catami"]},

                 ):
        self.api_key = api_key
        self.config = config

        if self.api_key == None:
            self.load_api_token()
        self.host = host
        self.id = int(config.annotations['annotation_set_id'])
        self.include_columns = include_columns
        self.template= template
        self.f = f
        self.q = q
        self.translate = translate
        self.api = None
        self.data = None
    def load_api_token(self,info_dir = './info'):
        with open(os.path.join(info_dir, 'API_TOKEN.txt'), "r") as file:
            api_key = file.read().strip()
            self.api_key =api_key
            cprint("API Token was loaded successfully ✅", "green")

        return api_key

    def load_api(self):
        self.api = SQAPI(host = self.host, api_key=self.api_key)
    def load_annotations(self, id = None, include_columns = None, filters = None):
        if self.api == None:
            self.load_api()
        if id == None:
            if self.id == 0:    
                id = input("Please enter the annotation set id:  ")
                self.id = id
            else:
                id = self.id
        if include_columns == None:
            include_columns = self.include_columns
        if filters == None:
            filters = self.q['filters']
        dict = self.api.export(f"/api/annotation_set/{id}/export",include_columns = include_columns, ).filter(name='label_id', op='is_not_null').execute().json()
        df = pd.json_normalize(dict.get('objects'))
        self.data = df

class annotation_set():
    def __init__(self, config,df):
        self.config = config
        self.df =df
        self.whole_frame_annotations = None
        self.point_annotations = None
    def get_whole_frame_annotations(self, df =None):
        if df == None:
            df = self.df

        if not {'label.name'}.issubset(df.columns):
            raise ValueError("DataFrame must contain 'label.name' ")
        if {'point.x','point.y'}.issubset(df.columns):
            filtered = df[df['point.x'].isna() & df['point.y'].isna()] #if there is labels with point.x,point.y, ignore them because there are
            #not wholeframe annotatoins
        else:
            filtered = df   # if point.x and point.y is not in the df, then assume all labels are wholeframe annotations

        unique_labels = filtered['label.name'].dropna().unique()
        whole_frame_annotations = list(unique_labels)
        self.whole_frame_annotations = whole_frame_annotations
        return whole_frame_annotations
    def get_point_annotations(self, df=None):
        if df is None:
            df = self.df
        if not {'point.x', 'point.y'}.issubset(df.columns):
            print("Dataframe does not contain point.x and point.y, therefore no point annotations")
            return []
        else:
            if not {'label.name'}.issubset(df.columns):
                raise ValueError("DataFrame must contain 'label.name'")
            
            # Select rows where both x and y are NOT null
            filtered = df[df['point.x'].notna() & df['point.y'].notna()]
            unique_labels = filtered['label.name'].dropna().unique()
            point_annotations = list(unique_labels)
            self.point_annotations = point_annotations
            return point_annotations






    def download_set(self, 
                    df=None,
                    dataset_name= None,
                    download_dir=None,
                    which_set=None,
                    suffix="_finetuning", 
                    override=False):
        
        if df is None:
            df = self.df
        print(df)
        if dataset_name == None:
            dataset_name = self.config.annotations['dataset_name']
        if download_dir == None:
            download_dir = self.config.annotations['download_dir']
        if which_set == None:
            which_set = self.config.annotations['which_set']
        print(f"df_filtered {len(df)}")
        print(f"Number of unique media IDs: {df['point.media.id'].nunique()}")

        self.get_whole_frame_annotations()
        self.get_point_annotations()

        if which_set == "whole_frame_annotations":
            labels_names = self.whole_frame_annotations
            if {'point.x', 'point.y'}.issubset(df.columns):
                df_filtered = df[df['point.x'].isna() & df['point.y'].isna()]
            else:
                df_filtered = df
        elif which_set == "point_annotations":
            labels_names = self.point_annotations
            df_filtered = df[df['point.x'].notna() & df['point.y'].notna()]
        else:
            raise ValueError("Invalid value for 'which_set'. Choose 'whole_frame_annotations' or 'point_annotations'.")

        os.makedirs(download_dir, exist_ok=True)

        label_dirs = {}
        for label in labels_names:
            label_dir = os.path.join(download_dir, dataset_name, which_set + suffix, label.replace("/", "_"))
            os.makedirs(label_dir, exist_ok=True)
            label_dirs[label] = label_dir

        print(f"df_filtered {len(df_filtered)}")
        df_valid = df_filtered[
            df_filtered['label.name'].isin(labels_names) & 
            df_filtered['point.media.path_best'].notna()
        ]
        print(f"Number of unique media IDs: {df['point.media.id'].nunique()}")
        print(f"Starting download of {len(df_valid)} images...")

        point_count = {}

        for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Downloading images"):
            label = row['label.name']
            image_url = row['point.media.path_best']
            filename = os.path.basename(image_url)
            save_path = os.path.join(label_dirs[label], filename)

            needs_download = override or not os.path.exists(save_path)

            if not needs_download:
                try:
                    img = Image.open(save_path)
                    img.verify()  # Check if corrupted
                except Exception:
                    print(f"Corrupted image detected. Will re-download: {save_path}")
                    needs_download = True

            if needs_download:
                try:
                    response = requests.get(image_url, stream=True, timeout=10)
                    if response.status_code == 200:
                        with open(save_path, 'wb') as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)
                    else:
                        print(f"Failed to download {image_url} — status code {response.status_code}")
                        continue
                except Exception as e:
                    print(f"Error downloading {image_url}: {e}")
                    continue

            if which_set == "point_annotations":
                try:
                    img = Image.open(save_path).convert("RGB")
                    img_width, img_height = img.size

                    x = int(float(row['point.x']) * img_width)
                    y = int(float(row['point.y']) * img_height)

                    left = max(min(x - 50, img_width - 100), 0)
                    upper = max(min(y - 50, img_height - 100), 0)
                    right = left + 100
                    lower = upper + 100

                    crop = img.crop((left, upper, right, lower))

                    base, ext = os.path.splitext(filename)
                    count = point_count.get(filename, 0) + 1
                    point_count[filename] = count
                    cropped_filename = f"{base}_p{count}{ext}"
                    cropped_path = os.path.join(label_dirs[label], cropped_filename)
                    crop.save(cropped_path)
                except Exception as e:
                    print(f"Cropping failed for {save_path}: {e}")
                    continue

        df_dir = os.path.join(download_dir, dataset_name, which_set + suffix, '_df.csv')
        df_filtered.to_csv(df_dir)
        print("Download and cropping complete." if which_set == "point_annotations" else "Download complete.")


class API_requests():
    def __init__(self, config):
        self.API_TOKEN = None
        self.ANNOTATIONS_URL = None
        self.downloads_dir = config.annotations['downloads_dir']
        self.annotations_dir = config.annotations['annotations_dir']
        self.utils = utils(config)
        if not os.path.exists(self.downloads_dir):
            os.makedirs(self.downloads_dir)
            print(f"Directory '{self.downloads_dir}' was created.")
        else:
            print(f"Directory '{self.downloads_dir}' already exists.")
    def download_annotations(self,API_TOKEN  =None, ANNOTATIONS_URL= None, override = False):
        if API_TOKEN is None:
            if self.API_TOKEN is not None:
                API_TOKEN = self.API_TOKEN
            else:
                print("No API Token was found, please load the API Token")
        if ANNOTATIONS_URL is None:
            if self.ANNOTATIONS_URL is not None:
                ANNOTATIONS_URL = self.ANNOTATIONS_URL
            else:
                print("No Annotation URL Token was found, please load the Annotation URL")
        downloads_dir = self.downloads_dir
        file_path = os.path.join(downloads_dir, 'annotations.csv')
        enable_download = True
        # Check if the file exists
        if os.path.exists(file_path) and override == False:
            print("Annotations file exists and loading it:", file_path)
            enable_download = False
        elif os.path.exists(file_path) and override == True:
            print("Annotations file exists but will override it")
        else:
            print("Annotations file does not exist:", file_path)
        if enable_download:
            headers = {"auth-token": API_TOKEN}

            # Step 1: Make the initial request
            with requests.Session() as s:
                response = s.get(ANNOTATIONS_URL, headers=headers)

            # Handle CSV response immediately
            if response.status_code == 200:
                print("Request successful! Downloading CSV...")
                with open("download/annotations.csv", "wb") as file:
                    file.write(response.content)
                print("CSV file saved successfully.")
                exit()

            elif response.status_code == 202:
                print("202 Accepted: Request is being processed in the background.")

                try:
                    response_json = response.json()
                    print("API Response:", response_json)  # Debugging
                except Exception as e:
                    print(f"Error: Expected JSON but got:\n{response.text}\nException: {e}")
                    exit()

                # Extract status_url and result_url correctly
                status_url = "https://www.squidle.org" + response_json.get("status_url", "")
                result_url = "https://www.squidle.org" + response_json.get("result_url", "")

                if not status_url or not result_url:
                    print("Error: API did not return valid status or result URLs.")
                    exit()

                print(f"Task accepted. Polling status at: {status_url}")

                # Step 3: Poll the status_url until task is completed
                while True:
                    status_response = requests.get(status_url, headers=headers)
                    
                    if status_response.status_code == 404:
                        print("Error: Task not found. The task might have expired.")
                        exit()

                    print(f"Raw response from status URL:\n{status_response.text}")  # Debugging

                    if status_response.status_code != 200:
                        print(f"Error: Received {status_response.status_code} from status URL.")
                        exit()

                    try:
                        status_data = status_response.json()
                    except Exception as e:
                        print(f"Error: Unable to parse JSON response. Possible HTML received.\n{e}")
                        exit()

                    # ✅ When task is done, download the result immediately
                    if status_data.get("status") == "done":
                        print("Task completed! Downloading the result...")

                        # Step 4: Download the result from result_url
                        result_response = requests.get(result_url, headers=headers)

                        if result_response.status_code == 200:
                            with open("downloads/annotations.csv", "wb") as file:
                                file.write(result_response.content)
                            print("✅ CSV file downloaded successfully as 'downloads/annotations.csv'.")
                            break
                        else:
                            print(f"❌ Failed to download result. Status code: {result_response.status_code}")

                        exit()  # ✅ Stop polling once the file is downloaded

                    elif status_data.get("status") == "failed":
                        print("❌ Task failed. Check API logs.")
                        exit()

                    print("Processing... Waiting 5 seconds before checking again.")
                    time.sleep(5)
                
            else:
                print(f"Error: Unexpected status code {response.status_code} - {response.text}")


    def Load_API(self,info_dir = './info'):
        with open(os.path.join(info_dir, 'API_TOKEN.txt'), "r") as file:
            API_TOKEN = file.read().strip()
            self.API_TOKEN =API_TOKEN
            cprint("API Token was loaded successfully ✅", "green")

        return API_TOKEN

    def Get_Annotations_URL(self,info_dir = './info'):
        try:
            with open(os.path.join(info_dir, 'ANNOTATIONS_URL.txt'), "r") as file:
                ANNOTATIONS_URL = file.read().strip()
                self.ANNOTATIONS_URL = ANNOTATIONS_URL
                cprint("Annotation URL was loaded successfully ✅", "green")
        except:
            cprint("Annotation url not found", "red")
            print("Please enter your annotation url:")
            # self.show_instruction_figure('figures/how_to_get_annotation_url.png')
            ANNOTATIONS_URL = input("Please enter the annotation set url> ")
            file_path = os.path.join(info_dir, 'ANNOTATIONS_URL.txt')

            self.utils.check_directory_exists(info_dir)
            
            ANNOTATIONS_URL = 'https://www.squidle.org' + ANNOTATIONS_URL
            with open(file_path, "w") as f:
                f.write(ANNOTATIONS_URL)
            cprint("Annotation URL was saved successfully ✅", "green")
        return ANNOTATIONS_URL

