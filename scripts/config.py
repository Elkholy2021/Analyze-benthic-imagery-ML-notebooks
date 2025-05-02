
import yaml




class Configuration_Classification:
    def __init__(self, config_path: str):
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        self.api_token = config['api_token']
        self.annotations = config['annotations']




