from visual_metrology.context import DPDContext


class DPDValidator:

    REQUIRED_FIELDS = [
        "metadata",
        "units",
        "features"
    ]

    def run(self, context: DPDContext) -> DPDContext:

        dpd = context.parsed_yaml

        if not isinstance(dpd, dict):
            raise ValueError("Invalid Digital Part Definition.")

        # ----------------------------------------------------------
        # Support both:
        #
        # metadata:
        # units:
        #
        # and
        #
        # dpd:
        #   metadata:
        #   units:
        # ----------------------------------------------------------

        if "dpd" in dpd:
            dpd = dpd["dpd"]

        missing = []

        for field in self.REQUIRED_FIELDS:

            if field not in dpd:
                missing.append(field)
                context.warning(f"Missing field : {field}")

        context.validation_report["missing_fields"] = missing

        context.validation_status = len(missing) == 0

        context.dpd = dpd

        context.log("DPD validation completed.")

        return context