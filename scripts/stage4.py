import os

import pandas as pd
from functions.stage4_immunogenicity_scoring import score_immunogenicity, plot_immunogenicity_scores

_TRUE_SPELLINGS = frozenset({"true", "1", "yes"})
_FALSE_SPELLINGS = frozenset({"false", "0", "no"})


def _parse_flag(name, value):
    """Read a boolean param, which Snakemake may deliver as a bool, an int or a str.

    `--config mc_dropout=false` arrives as the str 'false' (only the exact spellings
    True and False become bools), and so does a quoted YAML value, so bare bool()
    turned an explicit false into True. Anything that is not a recognised spelling
    raises rather than falling back to a default.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        spelling = value.lower()
        if spelling in _TRUE_SPELLINGS:
            return True
        if spelling in _FALSE_SPELLINGS:
            return False
    raise ValueError(
        f"{name}={value!r} is not a boolean: use true/false, yes/no or 1/0 in any case, "
        f"for example snakemake --config {name}=false"
    )


proteome_id = snakemake.wildcards.proteome_id
features = pd.read_csv(snakemake.input[0])
model_path = snakemake.params.get("model_path", None)
if model_path == "":
    model_path = None
# config.yaml's thresholds_path: the threshold picked for the configured model on the
# peptide-grouped OOF. Unset falls back to optimal_thresholds.json beside the model.
thresholds_path = snakemake.params.get("thresholds_path", None) or None
# calibration_path and mc_dropout reach score_immunogenicity only from here and from
# pipeline.py. Until this rule passed them, setting either in config.yaml was a silent
# no-op under Snakemake: calibration fell back to a calibrator beside the model, and
# mc_dropout=true produced no MC Dropout pass at all, though the ANN model card tells a
# reader to enable it exactly that way. Same defect as thresholds_path before it, and the
# mirror of tests/test_pipeline_forwards_freeze_mode.py, where pipeline.py was the side
# that dropped a key.
calibration_path = snakemake.params.get("calibration_path", None) or None
freeze_mode = _parse_flag("freeze_mode", snakemake.params.get("freeze_mode", False))
mc_dropout = _parse_flag("mc_dropout", snakemake.params.get("mc_dropout", False))
output_dir = os.path.dirname(str(snakemake.output[0])) or "."

ranked, model = score_immunogenicity(
    features,
    proteome_id,
    model_path=model_path,
    thresholds_path=thresholds_path,
    calibration_path=calibration_path,
    freeze_mode=freeze_mode,
    mc_dropout=mc_dropout,
    output_dir=output_dir,
)
plot_immunogenicity_scores(ranked, proteome_id, output_dir=output_dir)
