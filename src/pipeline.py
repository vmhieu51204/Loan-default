"""
End-to-End Scorecard Pipeline Orchestrator.
Encapsulates data ingestion, WoE binning, feature selection, logistic training,
isotonic calibration, scorecard scaling, evaluation, and master scale generation.
"""

from typing import Any, Dict, Optional, Tuple
import os
import pandas as pd
import numpy as np

from config.settings import ScorecardConfig
from src.data.loader import load_raw_data
from src.data.cleaner import clean_target_and_leakage
from src.features.engineering import (
    engineer_features,
    train_val_test_split_stratified,
)
from src.features.woe import WoEBinning
from src.features.selection import select_features
from src.models.train import train_logistic_model, save_model
from src.models.calibration import IsotonicProbabilityCalibrator
from src.scorecard.scaling import ScorecardScaler
from src.scorecard.master_scale import create_master_scale, create_decile_table
from src.evaluation.metrics import (
    evaluate_discrimination,
    evaluate_calibration_metrics,
)
from src.evaluation.plots import (
    plot_top_iv_features,
    plot_roc_curve,
    plot_score_distribution,
    plot_calibration_curve,
    plot_decile_calibration,
    plot_master_scale,
)


class ScorecardPipeline:
    """
    Production-grade credit scorecard pipeline coordinator.
    """

    def __init__(self, config: Optional[ScorecardConfig] = None):
        self.config = config or ScorecardConfig()
        self.woe_encoder = WoEBinning(
            max_bins=self.config.max_bins,
            min_samples=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )
        self.scaler = ScorecardScaler(
            pdo=self.config.pdo,
            base_score=self.config.base_score,
            base_odds=self.config.base_odds,
        )
        self.calibrator = IsotonicProbabilityCalibrator()

        # State storage
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_val: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None

        self.iv_df: Optional[pd.DataFrame] = None
        self.final_features: Optional[list] = None
        self.model = None

        self.test_results: Dict[str, Any] = {}

    def load_and_preprocess(
        self, filepath: Optional[str] = None, sample_size: Optional[int] = None
    ) -> None:
        """Loads data, removes leakages, engineers features, and partitions datasets."""
        data_file = filepath or self.config.data_path
        self.raw_df = load_raw_data(
            data_file,
            sample_size=sample_size,
            random_state=self.config.random_state,
        )

        cleaned = clean_target_and_leakage(
            self.raw_df,
            target_col=self.config.target_col,
            target_mapping=self.config.loan_status_mapping,
            leakage_cols=self.config.leakage_cols,
            binary_col_name=self.config.target_binary_col,
        )

        engineered = engineer_features(cleaned)
        self.cleaned_df = engineered

        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        ) = train_val_test_split_stratified(
            self.cleaned_df,
            target_col=self.config.target_binary_col,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_state=self.config.random_state,
        )

    def fit_woe_and_select_features(self) -> None:
        """Fits WoE on training set, transforms all partitions, and applies IV/Collinearity filters."""
        print("\n--- Fitting WoE Binning ---")
        self.woe_encoder.fit(self.X_train, self.y_train)

        # Transform all partitions
        self.X_train_woe = self.woe_encoder.transform(self.X_train)
        self.X_val_woe = self.woe_encoder.transform(self.X_val)
        self.X_test_woe = self.woe_encoder.transform(self.X_test)

        self.iv_df = self.woe_encoder.get_iv_summary()

        print("\n--- Performing Feature Selection ---")
        self.final_features, self.filtered_iv_df = select_features(
            self.X_train_woe,
            self.iv_df,
            min_iv=self.config.iv_min,
            max_iv=self.config.iv_max,
            corr_threshold=self.config.corr_threshold,
        )

    def train_and_calibrate(self) -> None:
        """Trains Logistic Regression on Train set and fits Isotonic Calibrator on Validation set."""
        print("\n--- Training Logistic Regression ---")
        X_train_final = self.X_train_woe[self.final_features]
        self.model = train_logistic_model(
            X_train_final,
            self.y_train,
            c_penalty=self.config.c_penalty,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
        )

        print("\n--- Calibrating Probabilities on Validation Partition ---")
        X_val_final = self.X_val_woe[self.final_features]
        y_val_raw_proba = self.model.predict_proba(X_val_final)[:, 1]
        self.calibrator.fit(y_val_raw_proba, self.y_val)

    def evaluate_and_scale(self) -> Dict[str, Any]:
        """Evaluates model performance on the hold-out Test set and generates scorecards."""
        print("\n--- Evaluating on Hold-out Test Set ---")
        X_test_final = self.X_test_woe[self.final_features].copy()

        # 1. Predictions
        y_test_raw_proba = self.model.predict_proba(X_test_final)[:, 1]
        y_test_calib_good_proba = self.calibrator.predict_good_proba(y_test_raw_proba)
        y_test_calib_pd = self.calibrator.predict_pd(y_test_raw_proba)

        # 2. Scorecard Scaling
        score_df = self.scaler.calculate_scores(
            X_test_final, self.model, self.final_features
        )
        test_scores = score_df["score"]

        # 3. Metrics
        disc_metrics = evaluate_discrimination(self.y_test, y_test_raw_proba)
        calib_metrics = evaluate_calibration_metrics(
            self.y_test, y_test_calib_good_proba
        )

        # 4. Master Scale & Decile Tables
        master_scale = create_master_scale(
            test_scores,
            y_test_calib_pd,
            min_score=self.config.master_scale_min_score,
            max_score=self.config.master_scale_max_score,
            bin_width=self.config.master_scale_bin_width,
        )

        decile_table = create_decile_table(test_scores, self.y_test, num_deciles=10)

        points_table = self.scaler.generate_scorecard_points_table(
            self.model, self.final_features, self.woe_encoder
        )

        self.test_results = {
            "discrimination_metrics": disc_metrics,
            "calibration_metrics": calib_metrics,
            "raw_proba": y_test_raw_proba,
            "calib_good_proba": y_test_calib_good_proba,
            "calib_pd": y_test_calib_pd,
            "scores": test_scores,
            "master_scale": master_scale,
            "decile_table": decile_table,
            "points_table": points_table,
        }

        return self.test_results

    def generate_plots(self, save_plots: bool = True, show_plots: bool = False) -> None:
        """Generates diagnostic and calibration visual charts."""
        fig_dir = os.path.join(self.config.output_dir, "figures") if save_plots else None

        plot_top_iv_features(
            self.iv_df.loc[self.final_features],
            top_n=15,
            save_path=os.path.join(fig_dir, "top_iv_features.png") if fig_dir else None,
            show=show_plots,
        )

        plot_roc_curve(
            self.test_results["discrimination_metrics"]["fpr"],
            self.test_results["discrimination_metrics"]["tpr"],
            self.test_results["discrimination_metrics"]["auc"],
            save_path=os.path.join(fig_dir, "roc_curve.png") if fig_dir else None,
            show=show_plots,
        )

        plot_score_distribution(
            self.test_results["scores"],
            self.y_test,
            save_path=os.path.join(fig_dir, "score_distribution.png") if fig_dir else None,
            show=show_plots,
        )

        plot_calibration_curve(
            self.y_test,
            self.test_results["raw_proba"],
            self.test_results["calib_good_proba"],
            save_path=os.path.join(fig_dir, "calibration_curve.png") if fig_dir else None,
            show=show_plots,
        )

        plot_decile_calibration(
            self.test_results["scores"],
            self.y_test,
            save_path=os.path.join(fig_dir, "decile_calibration.png") if fig_dir else None,
            show=show_plots,
        )

        plot_master_scale(
            self.test_results["master_scale"],
            save_path=os.path.join(fig_dir, "master_scale.png") if fig_dir else None,
            show=show_plots,
        )

    def save_artifacts(self) -> None:
        """Saves models, tables, and scorecard outputs to disk."""
        out_dir = self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)

        if self.model is not None:
            save_model(self.model, os.path.join(out_dir, "logistic_model.joblib"))

        if "points_table" in self.test_results:
            self.test_results["points_table"].to_csv(
                os.path.join(out_dir, "scorecard_points.csv"), index=False
            )
            print(f"Saved scorecard points table to: {os.path.join(out_dir, 'scorecard_points.csv')}")

        if "master_scale" in self.test_results:
            self.test_results["master_scale"].to_csv(
                os.path.join(out_dir, "master_scale.csv"), index=False
            )
            print(f"Saved master scale to: {os.path.join(out_dir, 'master_scale.csv')}")

        if "decile_table" in self.test_results:
            self.test_results["decile_table"].to_csv(
                os.path.join(out_dir, "decile_performance.csv"), index=False
            )
            print(f"Saved decile performance to: {os.path.join(out_dir, 'decile_performance.csv')}")

    def run(
        self,
        filepath: Optional[str] = None,
        sample_size: Optional[int] = None,
        save_plots: bool = True,
        show_plots: bool = False,
    ) -> Dict[str, Any]:
        """Runs the complete end-to-end scorecard development pipeline."""
        print("=" * 60)
        print("[PIPELINE] STARTING CREDIT SCORECARD DEVELOPMENT PIPELINE")
        print("=" * 60)

        self.load_and_preprocess(filepath=filepath, sample_size=sample_size)
        self.fit_woe_and_select_features()
        self.train_and_calibrate()
        results = self.evaluate_and_scale()
        self.generate_plots(save_plots=save_plots, show_plots=show_plots)
        self.save_artifacts()

        print("=" * 60)
        print("[SUCCESS] CREDIT SCORECARD PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        return results
