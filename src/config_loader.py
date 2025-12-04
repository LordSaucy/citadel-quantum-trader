class Config:
    _instance = None
    SETTINGS_PATH = "/opt/config/current_config.yaml"
    OPTIMISED_PATH = "/opt/config/new_config.yaml"
    FLAG_PATH = "/opt/config/use_optimised_cfg.flag"

    def _load(self):
        # Decide which file to read
        if os.path.exists(self.FLAG_PATH):
            path = self.OPTIMISED_PATH
        else:
            path = self.SETTINGS_PATH
        with open(path) as f:
            self.settings = yaml.safe_load(f)
