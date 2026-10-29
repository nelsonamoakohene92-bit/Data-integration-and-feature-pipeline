import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SPRINT_URL = (
    "https://raw.githubusercontent.com/RandulaKoralage/AgileScrumSprintVelocityDataSet/master/"
    "Agile%20Scrum%20Dataset/Finalized%20Datasets%20for%20Spring%20XD%20Project%20by%20Randula/"
    "Spring%20XD%20Sprints%2067.csv"
)
DEFECT_URL = "https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/main/kc1.csv"

SEED = 42
HORIZONS = (1, 2, 3)
LAGS = (1, 2, 3)
ROLL_WINDOWS = (3, 5)
TRAIN_SIZE = 46
VAL_SIZE = 10
TEST_SIZE = 10
MAX_REWORK = 0.9
LINK_VARIANTS = ("uniform", "scope_weighted", "ordered_blocks")

SPRINT_COUNT_COLUMNS = [
    "totalNumberOfIssues",
    "completedIssuesCount",
    "issuesNotCompletedInCurrentSprint",
    "puntedIssues",
    "issuesCompletedInAnotherSprint",
    "completedIssuesInitialEstimateSum",
    "completedIssuesEstimateSum",
    "puntedIssuesInitialEstimateSum",
    "puntedIssuesEstimateSum",
    "issuesNotCompletedInitialEstimateSum",
    "issuesNotCompletedEstimateSum",
    "issuesCompletedInAnotherSprintInitialEstimateSum",
    "issuesCompletedInAnotherSprintEstimateSum",
    "NoOfDevelopers",
    "SprintLength",
]
DATE_COLUMNS = ["sprintStartDate", "sprintEndDate", "sprintCompleteDate"]
DATE_FORMAT = "%d-%b-%y"


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    with urllib.request.urlopen(url, timeout=60) as response:
        target.write_bytes(response.read())
    return target


def load_sprints(path):
    frame = pd.read_csv(path)
    frame.columns = [c.strip() for c in frame.columns]
    for column in DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], format=DATE_FORMAT, errors="coerce")
    for column in SPRINT_COUNT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["sprintStartDate"])
    frame = frame.drop_duplicates(subset=["sprintId"])
    frame[SPRINT_COUNT_COLUMNS] = frame[SPRINT_COUNT_COLUMNS].fillna(0.0)
    frame["NoOfDevelopers"] = frame["NoOfDevelopers"].clip(lower=1.0)
    frame = frame.sort_values(["sprintStartDate", "sprintId"]).reset_index(drop=True)
    frame["sprint_index"] = np.arange(len(frame))
    return frame


def load_defects(path):
    frame = pd.read_csv(path)
    frame.columns = [c.strip() for c in frame.columns]
    label = frame["defects"].astype(str).str.strip().str.lower()
    frame["defects"] = label.map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
    feature_columns = [c for c in frame.columns if c != "defects"]
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().reset_index(drop=True)
    frame["defects"] = frame["defects"].astype(int)
    return frame


def link_defects_to_sprints(defects, sprints, variant, seed=SEED):
    count = len(sprints)
    size = len(defects)
    rng = np.random.default_rng(seed)
    if variant == "uniform":
        assignment = rng.integers(0, count, size=size)
    elif variant == "scope_weighted":
        weights = sprints["totalNumberOfIssues"].to_numpy(dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(count, 1.0 / count)
        assignment = rng.choice(count, size=size, p=weights)
    elif variant == "ordered_blocks":
        assignment = np.minimum((np.arange(size) * count) // size, count - 1)
    else:
        raise ValueError("unknown linkage variant: " + str(variant))
    linked = defects.copy()
    linked["sprint_index"] = assignment
    grouped = linked.groupby("sprint_index")["defects"].agg(modules="count", defective="sum")
    grouped = grouped.reindex(range(count), fill_value=0).reset_index()
    grouped["rework_share"] = np.where(
        grouped["modules"] > 0, grouped["defective"] / grouped["modules"].clip(lower=1), 0.0
    ).clip(0.0, MAX_REWORK)
    return grouped


def build_base_table(sprints, defect_summary):
    table = sprints.merge(defect_summary, on="sprint_index", how="left")
    table["rework_share"] = table["rework_share"].fillna(0.0)
    table["velocity"] = table["completedIssuesEstimateSum"]
    table["velocity_per_developer"] = table["velocity"] / table["NoOfDevelopers"]
    table["remaining_scope"] = table["issuesNotCompletedEstimateSum"] + table["puntedIssuesEstimateSum"]
    effective = table["velocity_per_developer"] * (1.0 - table["rework_share"])
    table["effective_velocity_per_developer"] = effective
    table["demand"] = np.where(effective > 0, table["remaining_scope"] / effective.where(effective > 0), np.nan)
    table["scope_change_ratio"] = np.where(
        table["totalNumberOfIssues"] > 0, table["puntedIssues"] / table["totalNumberOfIssues"].clip(lower=1), 0.0
    )
    table["completion_ratio"] = np.where(
        table["totalNumberOfIssues"] > 0,
        table["completedIssuesCount"] / table["totalNumberOfIssues"].clip(lower=1),
        0.0,
    )
    return table


def add_features(table):
    out = table.copy()
    base_columns = [
        "velocity",
        "velocity_per_developer",
        "rework_share",
        "remaining_scope",
        "demand",
        "NoOfDevelopers",
    ]
    for column in base_columns:
        for lag in LAGS:
            out[column + "_lag" + str(lag)] = out[column].shift(lag)
    for column in ("velocity_per_developer", "rework_share", "demand"):
        for window in ROLL_WINDOWS:
            shifted = out[column].shift(1)
            out[column + "_roll_mean" + str(window)] = shifted.rolling(window, min_periods=window).mean()
            out[column + "_roll_std" + str(window)] = shifted.rolling(window, min_periods=window).std()
    out["rework_trend"] = out["rework_share"].shift(1) - out["rework_share"].shift(2)
    out["velocity_trend"] = out["velocity_per_developer"].shift(1) - out["velocity_per_developer"].shift(2)
    out["scope_change_lag1"] = out["scope_change_ratio"].shift(1)
    out["completion_ratio_lag1"] = out["completion_ratio"].shift(1)
    out["month"] = out["sprintStartDate"].dt.month
    for horizon in HORIZONS:
        out["target_h" + str(horizon)] = out["demand"].shift(-horizon)
    return out


def feature_columns(frame):
    excluded_prefixes = ("target_h",)
    excluded = {
        "sprintId",
        "sprintName",
        "sprintState",
        "sprintStartDate",
        "sprintEndDate",
        "sprintCompleteDate",
        "sprint_index",
        "split",
    }
    columns = []
    for column in frame.columns:
        if column in excluded or column.startswith(excluded_prefixes):
            continue
        if column.endswith("_lag1") or "_lag" in column or "_roll_" in column or column in (
            "rework_trend",
            "velocity_trend",
            "month",
        ):
            columns.append(column)
    return columns


def assign_splits(frame):
    total = len(frame)
    needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if total < needed:
        raise ValueError("not enough sprints for the requested split")
    labels = np.array(["unused"] * total, dtype=object)
    labels[:TRAIN_SIZE] = "train"
    labels[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE] = "validation"
    labels[total - TEST_SIZE :] = "test"
    out = frame.copy()
    out["split"] = labels
    return out


def purge_overlap(frame):
    out = frame.copy()
    max_horizon = max(HORIZONS)
    order = out["sprint_index"].to_numpy()
    boundaries = {}
    for name in ("train", "validation", "test"):
        indices = out.loc[out["split"] == name, "sprint_index"]
        if len(indices):
            boundaries[name] = (int(indices.min()), int(indices.max()))
    keep = np.ones(len(out), dtype=bool)
    for name, (low, high) in boundaries.items():
        mask = (out["split"] == name).to_numpy()
        reach = order + max_horizon
        keep &= ~(mask & (reach > high))
    return out.loc[keep].reset_index(drop=True)


def finalise_splits(frame):
    columns = feature_columns(frame)
    targets = ["target_h" + str(h) for h in HORIZONS]
    labelled = assign_splits(frame)
    labelled = labelled[labelled["split"] != "unused"]
    labelled = purge_overlap(labelled)
    labelled = labelled.dropna(subset=columns + targets).reset_index(drop=True)
    return labelled, columns, targets


def scale_features(frame, columns):
    train_mask = frame["split"] == "train"
    scaler = StandardScaler()
    scaler.fit(frame.loc[train_mask, columns])
    scaled = frame.copy()
    scaled[columns] = scaler.transform(frame[columns])
    return scaled, scaler


def validate(frame, columns, targets):
    problems = []
    for name in ("train", "validation", "test"):
        if not (frame["split"] == name).any():
            problems.append("empty split: " + name)
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"]
    if len(train) and len(val) and train["sprint_index"].max() + max(HORIZONS) >= val["sprint_index"].min():
        problems.append("train targets overlap validation window")
    if len(val) and len(test) and val["sprint_index"].max() + max(HORIZONS) >= test["sprint_index"].min():
        problems.append("validation targets overlap test window")
    if len(train) and len(val) and train["sprintStartDate"].max() > val["sprintStartDate"].min():
        problems.append("train dates exceed validation dates")
    if len(val) and len(test) and val["sprintStartDate"].max() > test["sprintStartDate"].min():
        problems.append("validation dates exceed test dates")
    if frame[columns + targets].isna().any().any():
        problems.append("missing values remain")
    if not np.isfinite(frame[columns + targets].to_numpy(dtype=float)).all():
        problems.append("non finite values remain")
    if (frame[targets] < 0).any().any():
        problems.append("negative demand targets")
    leaking = [c for c in columns if c.startswith("target_h") or c == "demand"]
    if leaking:
        problems.append("target columns inside features: " + ", ".join(leaking))
    if frame["sprint_index"].duplicated().any():
        problems.append("duplicate sprint rows")
    return problems


def run_variant(sprints, defects, variant, seed, output_dir):
    summary = link_defects_to_sprints(defects, sprints, variant, seed)
    base = build_base_table(sprints, summary)
    featured = add_features(base)
    splits, columns, targets = finalise_splits(featured)
    scaled, scaler = scale_features(splits, columns)
    problems = validate(scaled, columns, targets)
    folder = Path(output_dir) / variant
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        part = scaled[scaled["split"] == name]
        part.to_csv(folder / (name + "_scaled.csv"), index=False)
        splits[splits["split"] == name].to_csv(folder / (name + "_raw.csv"), index=False)
    base.to_csv(folder / "sprint_table.csv", index=False)
    meta = {
        "variant": variant,
        "seed": seed,
        "feature_columns": columns,
        "target_columns": targets,
        "rows": {n: int((scaled["split"] == n).sum()) for n in ("train", "validation", "test")},
        "scaler_mean": dict(zip(columns, map(float, scaler.mean_))),
        "scaler_scale": dict(zip(columns, map(float, scaler.scale_))),
        "problems": problems,
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta, base


def describe(meta, base):
    print("variant:", meta["variant"])
    print("rows:", meta["rows"])
    print("features:", len(meta["feature_columns"]))
    print("mean rework share:", round(float(base["rework_share"].mean()), 4))
    print("median demand:", round(float(base["demand"].median()), 4))
    print("problems:", meta["problems"] if meta["problems"] else "none")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--variant", default="all", choices=("all",) + LINK_VARIANTS)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    raw = root / "data" / "raw"
    out = root / "data" / "processed"
    sprint_path = raw / "sprints.csv"
    defect_path = raw / "kc1.csv"
    if not args.offline:
        download(SPRINT_URL, sprint_path)
        download(DEFECT_URL, defect_path)
    sprints = load_sprints(sprint_path)
    defects = load_defects(defect_path)
    print("sprints:", len(sprints), "defect modules:", len(defects))
    variants = LINK_VARIANTS if args.variant == "all" else (args.variant,)
    failed = False
    for variant in variants:
        meta, base = run_variant(sprints, defects, variant, args.seed, out)
        describe(meta, base)
        failed = failed or bool(meta["problems"])
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
