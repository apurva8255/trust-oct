import os
import yaml

DEFAULT_CONFIG = {
    "paths": {
        "raw_data_dir": "./data/raw",
        "processed_data_dir": "./data/processed",
        "generated_qa_dir": "./data/generated_qa",
        "reports_dir": "./reports"
    },
    "generation": {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": 1000
    },
    "evaluation": {
        "default_model": "gpt-4o",
        "batch_size": 5,
        "output_format": "csv"
    },
    "visualization": {
        "bbox_colors": {
            "IRF": "purple",
            "SRF": "blue",
            "PED": "red",
            "SHRM": "orange",
            "MH": "green",
            "default": "yellow"
        },
        "line_width": 3
    }
}

class Config:
    def __init__(self, config_path="config.yaml"):
        self.config_data = DEFAULT_CONFIG.copy()
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        # Deep merge configs
                        for key in user_config:
                            if key in self.config_data and isinstance(self.config_data[key], dict):
                                self.config_data[key].update(user_config[key])
                            else:
                                self.config_data[key] = user_config[key]
            except Exception as e:
                print(f"Warning: Failed to load config.yaml ({e}). Using defaults.")
        
        # Expose config keys as attributes
        self.paths = self.config_data.get("paths", {})
        self.generation = self.config_data.get("generation", {})
        self.evaluation = self.config_data.get("evaluation", {})
        self.visualization = self.config_data.get("visualization", {})

        # Ensure folders exist
        for folder_path in self.paths.values():
            os.makedirs(folder_path, exist_ok=True)

# Global configuration instance
config = Config()
