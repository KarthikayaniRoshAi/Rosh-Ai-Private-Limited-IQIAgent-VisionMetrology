from pathlib import Path

import yaml

from visual_metrology.context import DPDContext
from visual_metrology.train.drawing_intelligence.prompt import SYSTEM_PROMPT


class DPDExporter:

    def run(self, context: DPDContext, output_config: dict) -> DPDContext:

        base_output = Path(output_config["directory"])

        if output_config["create_drawing_folder"]:
            output_dir = base_output / context.drawing_path.stem
        else:
            output_dir = base_output

        output_dir.mkdir(parents=True, exist_ok=True)

        drawing_name = context.drawing_path.stem

        # --------------------------------------------------
        # Raw GPT Response
        # --------------------------------------------------

        raw_response_file = output_dir / "raw_gpt_response.txt"

        with open(raw_response_file, "w", encoding="utf-8") as f:
            f.write(context.raw_response)

        context.metadata["raw_response_file"] = str(raw_response_file)

        # --------------------------------------------------
        # Raw YAML
        # --------------------------------------------------

        raw_yaml_file = output_dir / "raw_yaml.yaml"

        with open(raw_yaml_file, "w", encoding="utf-8") as f:
            f.write(context.raw_yaml)

        context.metadata["raw_yaml_file"] = str(raw_yaml_file)

        # --------------------------------------------------
        # Parsed YAML
        # --------------------------------------------------

        parsed_yaml_file = output_dir / "parsed.yaml"

        with open(parsed_yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(
                context.parsed_yaml,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )

        context.metadata["parsed_yaml_file"] = str(parsed_yaml_file)

        # --------------------------------------------------
        # Validation Report
        # --------------------------------------------------

        validation_file = output_dir / "validation_report.txt"

        with open(validation_file, "w", encoding="utf-8") as f:

            f.write("=" * 60 + "\n")
            f.write("Digital Part Definition Validation Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Drawing : {context.drawing_name}\n")
            f.write(f"Status  : {'PASSED' if context.validation_status else 'FAILED'}\n\n")

            if context.validation_report.get("missing_fields"):

                f.write("Missing Fields\n")
                f.write("-" * 40 + "\n")

                for field in context.validation_report["missing_fields"]:
                    f.write(f"- {field}\n")

                f.write("\n")

            if context.warnings:

                f.write("Warnings\n")
                f.write("-" * 40 + "\n")

                for warning in context.warnings:
                    f.write(f"- {warning}\n")

                f.write("\n")

            if context.errors:

                f.write("Errors\n")
                f.write("-" * 40 + "\n")

                for error in context.errors:
                    f.write(f"- {error}\n")

        drawing_name = context.drawing_path.stem

        if output_config["save_raw_yaml"]:

            raw_yaml_file = output_dir / f"{drawing_name}_raw.yaml"

            with open(raw_yaml_file, "w", encoding="utf-8") as f:
                f.write(context.raw_yaml)

            context.metadata["raw_yaml_file"] = str(raw_yaml_file)

        if output_config["save_dpd_yaml"]:

            dpd_file = output_dir / f"{drawing_name}_drawing_analysis.yaml"

            with open(dpd_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    context.dpd,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True
                )

            context.metadata["dpd_file"] = str(dpd_file)

        if output_config["save_prompt"]:

            prompt_file = output_dir / "prompt.txt"

            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(SYSTEM_PROMPT)

            context.metadata["prompt_file"] = str(prompt_file)

        context.log("Digital Part Definition exported.")

        return context