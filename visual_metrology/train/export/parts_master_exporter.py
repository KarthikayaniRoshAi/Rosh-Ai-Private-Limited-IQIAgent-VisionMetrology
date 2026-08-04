from pathlib import Path
import yaml

from visual_metrology.context import DPDContext


class PartsMasterExporter:

    def run(
        self,
        context: DPDContext,
        output_config: dict
    ) -> DPDContext:

        base_output = Path(output_config["directory"])

        master_file = base_output / "parts_master.yaml"

        # -----------------------------------------
        # Load existing catalog
        # -----------------------------------------

        if master_file.exists():

            with open(master_file, "r", encoding="utf-8") as f:
                catalog = yaml.safe_load(f) or {}

        else:

            catalog = {}

        catalog.setdefault("parts", {})

        metadata = context.parsed_yaml.get("metadata", {})

        part_number = metadata.get("item_number", "").strip()

        if not part_number:
            context.warning(
                "Part number not found. Skipping parts_master update."
            )
            return context

        catalog["parts"][part_number] = {

            "description":
                metadata.get("description", ""),

            "revision":
                metadata.get("revision", ""),

            "date":
                metadata.get("date", ""),

            "originator":
                metadata.get("originator", "")
        }

        # -----------------------------------------
        # Sort by Part Number
        # -----------------------------------------

        catalog["parts"] = dict(
            sorted(catalog["parts"].items())
        )

        with open(master_file, "w", encoding="utf-8") as f:

            yaml.dump(
                catalog,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True
            )

        context.metadata["parts_master"] = str(master_file)

        return context