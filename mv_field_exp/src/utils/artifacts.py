"""Artifact naming and schema notes for law-to-law experiment outputs."""

RESULT_ARTIFACT = "results.json"
RESULT_TABLE_ARTIFACT = "results.csv"
BEST_METRICS_ARTIFACT = "best_metrics.json"
EVAL_BY_SPLIT_ARTIFACT = "eval_by_split.csv"
TRAINING_HISTORY_CSV = "training_history.csv"
EMPIRICAL_GAUSSIAN_ORACLE_FLOOR_JSON = "empirical_gaussian_oracle_floor.json"
CONTEXT_SENSITIVITY_JSON = "context_sensitivity.json"
CONTEXT_SENSITIVITY_CSV = "context_sensitivity.csv"
GENERATION_MANIFEST_ARTIFACT = "generation_manifest.json"
GENERATION_MANIFEST_COPY_YAML = "generation_manifest.yaml"
GENERATION_MANIFEST_YAML = "manifest.yaml"
GENERATION_SUMMARY_YAML = "generation_summary.yaml"
ATOMIZATION_SUMMARY_YAML = "atomization_summary.yaml"
ATOMIZATION_MANIFEST_COPY_YAML = "atomization_manifest.yaml"
SOURCE_DYNAMICS_MANIFEST_COPY_YAML = "source_dynamics_manifest.yaml"
CALIBRATION_SUMMARY_YAML = "calibration_summary.yaml"
CALIBRATION_CURVES_PT = "calibration_curves.pt"
PREDICTION_CURVES_PT = "prediction_curves.pt"
REFERENCE_EVAL_JSON = "reference_eval.json"
REFERENCE_EVAL_CSV = "reference_eval.csv"
REFERENCE_DENSITY_CURVES_PT = "reference_density_curves.pt"
REFERENCE_PREDICTION_CURVES_PT = "reference_prediction_curves.pt"
SELECTED_DENSITY_CURVES_PT = "selected_density_curves.pt"
TARGET_MAP_METADATA_YAML = "target_map_metadata.yaml"
BASIS_METADATA_YAML = "basis_metadata.yaml"
PROJECTION_SUMMARY_YAML = "projection_summary.yaml"
QA_SUMMARY_YAML = "qa_summary.yaml"
COMPONENT_COUNT_HIST_PNG = "component_count_hist.png"
INPUT_LAW_MOMENTS_PNG = "input_law_moments.png"
TARGET_MEAN_HIST_PNG = "target_mean_hist.png"
TARGET_COV_EIGS_PNG = "target_cov_eigs.png"
FEATURE_SUMMARY_PNG = "feature_summary.png"
LAW_FEATURE_PCA_PNG = "law_feature_pca.png"
SELECTED_INPUT_LAW_PARTICLES_PCA_PNG = "selected_input_law_particles_pca.png"
SELECTED_TARGET_LAW_PARTICLES_PCA_PNG = "selected_target_law_particles_pca.png"
SELECTED_LAW_DISTANCE_MATRIX_PNG = "selected_law_distance_matrix.png"
QA_PNG_ARTIFACTS = [
    COMPONENT_COUNT_HIST_PNG,
    INPUT_LAW_MOMENTS_PNG,
    TARGET_MEAN_HIST_PNG,
    TARGET_COV_EIGS_PNG,
    FEATURE_SUMMARY_PNG,
    LAW_FEATURE_PCA_PNG,
    SELECTED_INPUT_LAW_PARTICLES_PCA_PNG,
    SELECTED_TARGET_LAW_PARTICLES_PCA_PNG,
    SELECTED_LAW_DISTANCE_MATRIX_PNG,
]
