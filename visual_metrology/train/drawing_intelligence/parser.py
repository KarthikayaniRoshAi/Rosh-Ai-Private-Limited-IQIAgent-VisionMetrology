import yaml

from visual_metrology.context import DPDContext


class YAMLParser:

    def run(self, context: DPDContext) -> DPDContext:
        if not context.raw_yaml.strip():
            raise ValueError("GPT response is empty.")

        try:
            context.parsed_yaml = yaml.safe_load(context.raw_yaml)
            context.log("YAML parsed successfully.")
        except yaml.YAMLError as e:
            context.error(str(e))
            raise

        return context