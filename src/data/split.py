"""
src/data/split.py

Split train/val/test BY video_id (not by clip/frame), stratified by label
to keep the accident/normal ratio stable across all sets.

Why splitting by video_id is mandatory:
- Frames/clips cut from the SAME video share background, camera angle,
  lighting, etc. If we split randomly by clip/frame, it's very easy for
  part of video A to end up in train and another part in test.
- In that case, the model doesn't need to learn real "accident features" -
  it just needs to "recognize" video A from training to predict correctly
  on the test portion.
- Result: test accuracy/F1 gets artificially inflated (leakage), not a
  true reflection of real generalization. This is a serious bug, easy for
  reviewers/recruiters to catch when they ask "how did you split the data?".

This module:
1. Takes a manifest (list of video_id + label).
2. Groups by video_id (each video_id appears EXACTLY once in the result).
3. Stratified split by label (train/val/test).
4. Asserts no overlap between the 3 sets (automatic leakage check).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_by_video_id(
    manifest_df: pd.DataFrame,
    video_id_col: str = "video_id",
    label_col: str = "label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    manifest_df: each row is 1 video_id (or 1 clip, which will be grouped
                 by video_id). If a video has multiple clips/multiple labels
                 (e.g. it has both an accident segment and a normal
                 segment), that video is assigned label = "accident" when
                 grouping (positive label takes priority), to guarantee the
                 whole video_id always stays in a single split.
    Returns: DataFrame [video_id, label, split] with split in {train,val,test}.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test ratio must sum to 1.0"

    # Aggregate to video_id level: if a video has both accident and normal
    # clips, prioritize the "accident" label for the whole video when
    # stratify-splitting, since this class is rarer and more important to
    # preserve the ratio for.
    def _agg_label(labels: pd.Series) -> str:
        return "accident" if (labels == "accident").any() else labels.iloc[0]

    video_level = (
        manifest_df.groupby(video_id_col)[label_col]
        .apply(_agg_label)
        .reset_index()
    )

    # Check if there are enough (>= 2) samples per class to stratify
    class_counts = video_level[label_col].value_counts()
    can_stratify = (class_counts >= 2).all() and len(class_counts) > 1

    train_ids, temp_ids = train_test_split(
        video_level,
        train_size=train_ratio,
        random_state=random_state,
        stratify=video_level[label_col] if can_stratify else None,
    )

    remaining_ratio = val_ratio / (val_ratio + test_ratio)
    can_stratify_temp = can_stratify and (temp_ids[label_col].value_counts() >= 2).all()
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=remaining_ratio,
        random_state=random_state,
        stratify=temp_ids[label_col] if can_stratify_temp else None,
    )

    train_ids = train_ids.copy(); train_ids["split"] = "train"
    val_ids = val_ids.copy(); val_ids["split"] = "val"
    test_ids = test_ids.copy(); test_ids["split"] = "test"

    result = pd.concat([train_ids, val_ids, test_ids], ignore_index=True)
    _assert_no_leakage(result, video_id_col)
    _log_split_summary(result, label_col)
    return result


def _assert_no_leakage(split_df: pd.DataFrame, video_id_col: str) -> None:
    """Check that no video_id appears in more than 1 split. Fail fast if it does."""
    dup = split_df[split_df.duplicated(subset=[video_id_col], keep=False)]
    if len(dup) > 0:
        raise ValueError(
            f"LEAKAGE DETECTED: {dup[video_id_col].nunique()} video_id values appear "
            f"in more than 1 split. Example: {dup[video_id_col].unique()[:5].tolist()}"
        )
    logger.info("Leakage check: OK - each video_id belongs to exactly 1 split.")


def _log_split_summary(split_df: pd.DataFrame, label_col: str) -> None:
    summary = split_df.groupby(["split", label_col]).size().unstack(fill_value=0)
    logger.info(f"Split distribution:\n{summary}")


def apply_split_to_manifest(
    manifest_df: pd.DataFrame,
    split_map_df: pd.DataFrame,
    video_id_col: str = "video_id",
) -> pd.DataFrame:
    """Attach the 'split' column to a detailed manifest (clip/frame level)
    based on the split already computed at the video_id level."""
    merged = manifest_df.merge(
        split_map_df[[video_id_col, "split"]], on=video_id_col, how="left"
    )
    n_missing = merged["split"].isna().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} rows have no matching split (video_id mismatch) - will be dropped.")
    return merged.dropna(subset=["split"])


def save_split(split_df: pd.DataFrame, out_csv_path: str, out_json_path: Optional[str] = None) -> None:
    split_df.to_csv(out_csv_path, index=False)
    logger.info(f"Saved split map: {out_csv_path}")
    if out_json_path:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(split_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)