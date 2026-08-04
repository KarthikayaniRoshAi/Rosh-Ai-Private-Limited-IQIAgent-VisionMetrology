import yaml


class ConfigLoader:

    def __init__(self, config_file):

        with open(config_file, "r") as f:

            self.config = yaml.safe_load(f)


    def get(self):

        return self.config