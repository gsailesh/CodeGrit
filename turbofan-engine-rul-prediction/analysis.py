import os
import math
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt


def read_data(
    file_path: Path, file_name: str, columns: list[str]
) -> tuple[pl.DataFrame, str]:
    """Read dataset file from specified directory"""
    suffix = file_name.split("_")[-1].split(".")[0]

    if file_name.startswith("train_"):
        suffix = "train_" + suffix
        return (
            pl.read_csv(
                file_path,
                has_header=False,
                separator=" ",
                new_columns=columns,
            ),
            suffix,
        )
    elif file_name.startswith("test_"):
        suffix = "test_" + suffix
        return (
            pl.read_csv(
                file_path,
                has_header=False,
                separator=" ",
                new_columns=columns,
            ),
            suffix,
        )
    elif file_name.startswith("RUL_"):
        return (
            pl.read_csv(
                file_path,
                has_header=False,
                separator=" ",
                new_columns=["RUL"],
            ),
            suffix,
        )


def read_dataset(
    dataset_dir: Path, inclusions: list[str], exclusions: list[str], columns: list[str]
) -> tuple[pl.DataFrame]:
    """Read train and test data from dataset directory."""
    files = os.listdir(dataset_dir)
    df_train = pl.DataFrame()
    df_test = pl.DataFrame()
    df_rul = pl.DataFrame()

    for file_ in files:
        if file_.endswith(".txt") and any(
            file_.startswith(inclusion) for inclusion in inclusions
        ):
            print(f"Reading file: {file_}")
            file_path = dataset_dir / file_
            df, suffix = read_data(file_path, file_, columns)
            if suffix.startswith("train"):
                df = df.with_columns(pl.lit(suffix).alias("desc"))
                df_train = df_train.vstack(df)
            elif suffix.startswith("test"):
                df = df.with_columns(pl.lit(suffix).alias("desc"))
                df_test = df_test.vstack(df)
            elif suffix.startswith("RUL"):
                df = df.with_columns(pl.lit(suffix).alias("desc"))
                df_rul = df_rul.vstack(df)

    df_train = df_train.drop(exclusions)
    df_test = df_test.drop(exclusions)

    return (df_train, df_test, df_rul)


def simple_data_checks(df: pl.DataFrame):
    """Simple print statements to understand the data a bit."""

    # 1. How many unique engines are there in all?
    print(
        "Unique engines in train file:",
        df.group_by(["desc"]).agg(pl.col("unit number").n_unique()),
    )
    # 2. What is the minimum and maximum life in cycle counts across all the engines?
    print(
        "Train min max cycle time:",
        df.group_by("desc").agg(
            pl.col("min_cycle_time").min(),
            pl.col("max_cycle_time").max(),
        ),
    )
    print(
        df[
            [
                "unit number",
                "time, in cycles",
                "min_cycle_time",
                "max_cycle_time",
                "rul",
            ]
        ].head()
    )

    print(
        df[
            [
                "unit number",
                "time, in cycles",
                "min_cycle_time",
                "max_cycle_time",
                "rul",
            ]
        ].filter((pl.col("unit number") == 1) & (pl.col("time, in cycles") == 100))
    )


def basic_data_quality_checks(df: pl.DataFrame):
    """Do the basic data quality checks to the Polars DataFrame."""

    ## 1. Check for dead sensors - sensors that have zero variance across the dataset.
    sensor_cols = [c for c in df.columns if c.startswith("sensor measurement")]
    dead_sensors = []
    print("Before: ", df.shape)
    for col in sensor_cols:
        sensor_std_by_engine = df.group_by(["desc"]).agg(pl.col(col).std())
        print(
            f"Std: {col} is ",
            sensor_std_by_engine.head(5),
        )

        if (sensor_std_by_engine[col] < 1e-2).any():
            print(f"Column {col} has zero standard deviation.")
            dead_sensors.append(col)

    print(f"Dead sensors: {dead_sensors}")
    df = df.drop(dead_sensors)

    ## 2. Missing value checks
    df = df.drop_nulls()
    df = df.drop_nans()
    print("After: ", df.shape)
    return df


def plot_single_engine_lifecycle(
    df_engine: pl.DataFrame, sensor_col: list[str], time_col: list[str]
):
    """Create a grid of subplots for all the sensors for a single engine."""
    plt.figure(figsize=(18, 24))
    for i, sensor in enumerate(sensor_col):
        plt.subplot(math.ceil(len(sensor_col) / 3), 3, i + 1)
        plt.plot(df_engine[time_col], df_engine[sensor])
        plt.title(sensor)
        plt.xlabel("Time in Cycles")
        plt.ylabel("Sensor Measurement")
    plt.tight_layout()
    plt.show()


def plot_multiple_engine_lifecycle(
    df_engines: pl.DataFrame, sensor_col: list[str], time_col: str, n_engines: int = 5
):
    """Overlay multiple engines on the same subplot grid."""
    fig, axes = plt.subplots(math.ceil(len(sensor_col) / 3), 3, figsize=(18, 24))
    axes = axes.flatten()

    engine_ids = df_engines["unit number"].unique().sort().head(n_engines).to_list()

    for engine_id in engine_ids:
        df_engine = df_engines.filter(pl.col("unit number") == engine_id)
        for i, sensor in enumerate(sensor_col):
            axes[i].plot(
                df_engine[time_col].to_list(),
                df_engine[sensor].to_list(),
                alpha=0.5,
                label=f"Engine {engine_id}",
            )
            axes[i].set_title(sensor)

    axes[0].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("multi_engine_lifecycle.png")


def main():
    data_dir = Path(
        "/Users/gsailesh/.cache/kagglehub/datasets/behrad3d/nasa-cmaps/versions/1"
    )
    cmaps_dir = data_dir / "CMaps"
    # print("files", files)

    inclusions = ["train_", "test_", "RUL_"]
    exclusions = ["column_27", "column_28"]

    train_test_columns = [
        "unit number",
        "time, in cycles",
        "operational setting 1",
        "operational setting 2",
        "operational setting 3",
        "sensor measurement 1",
        "sensor measurement 2",
        "sensor measurement 3",
        "sensor measurement 4",
        "sensor measurement 5",
        "sensor measurement 6",
        "sensor measurement 7",
        "sensor measurement 8",
        "sensor measurement 9",
        "sensor measurement 10",
        "sensor measurement 11",
        "sensor measurement 12",
        "sensor measurement 13",
        "sensor measurement 14",
        "sensor measurement 15",
        "sensor measurement 16",
        "sensor measurement 17",
        "sensor measurement 18",
        "sensor measurement 19",
        "sensor measurement 20",
        "sensor measurement 21",
    ]
    df_train, df_test, df_rul = read_dataset(
        cmaps_dir, inclusions, exclusions, train_test_columns
    )

    max_cycle_time = pl.DataFrame(
        df_train.group_by("unit number").agg(pl.col("time, in cycles").max()),
    )
    max_cycle_time.columns = ["unit number", "max_cycle_time"]
    min_cycle_time = pl.DataFrame(
        df_train.group_by("unit number").agg(pl.col("time, in cycles").min()),
    )
    min_cycle_time.columns = ["unit number", "min_cycle_time"]

    df_train = df_train.join(
        max_cycle_time,
        on="unit number",
    )
    df_train = df_train.join(min_cycle_time, on="unit number")

    df_train = df_train.with_columns(
        rul=pl.col("max_cycle_time") - pl.col("time, in cycles")
    )

    df_train = df_train.with_columns(
        rul=pl.col("rul").clip(lower_bound=0, upper_bound=125)
    )

    max_cycle_time = pl.DataFrame(
        df_test.group_by("unit number").agg(pl.col("time, in cycles").max()),
    )
    max_cycle_time.columns = ["unit number", "max_cycle_time"]
    min_cycle_time = pl.DataFrame(
        df_test.group_by("unit number").agg(pl.col("time, in cycles").min()),
    )
    min_cycle_time.columns = ["unit number", "min_cycle_time"]

    df_test = df_test.join(
        max_cycle_time,
        on="unit number",
    )
    df_test = df_test.join(min_cycle_time, on="unit number")

    df_test = df_test.with_columns(
        rul=pl.col("max_cycle_time") - pl.col("time, in cycles")
    )

    df_test = df_test.with_columns(
        rul=pl.col("rul").clip(lower_bound=0, upper_bound=125)
    )

    ## ----- Do further steps separately for each datasubset (ex: FD001, FD002, FD003, FD004). ----- ##
    df_train_fd001 = df_train.filter(pl.col("desc") == "train_FD001")
    df_train_fd002 = df_train.filter(pl.col("desc") == "train_FD002")
    df_train_fd003 = df_train.filter(pl.col("desc") == "train_FD003")
    df_train_fd004 = df_train.filter(pl.col("desc") == "train_FD004")

    df_test_fd001 = df_test.filter(pl.col("desc") == "test_FD001")
    df_test_fd002 = df_test.filter(pl.col("desc") == "test_FD002")
    df_test_fd003 = df_test.filter(pl.col("desc") == "test_FD003")
    df_test_fd004 = df_test.filter(pl.col("desc") == "test_FD004")

    simple_data_checks(df_train_fd001)
    simple_data_checks(df_test_fd001)

    df_train_fd001 = basic_data_quality_checks(df_train_fd001)
    df_test_fd001 = basic_data_quality_checks(df_test_fd001)

    ## Check RUL distribution - Are most engines short-lived or long-lived?
    engine_lifetimes = df_train_fd001.group_by("unit number").agg(
        pl.col("max_cycle_time").max()
    )
    rul_dist = df_train_fd001.group_by("unit number").agg(pl.col("rul").min())

    plt.hist(engine_lifetimes["max_cycle_time"].to_list(), bins=20, edgecolor="black")
    plt.xlabel("Max Cycles crossed before Failure")
    plt.ylabel("Number of Engines")
    plt.title("Engine Lifetime Distribution")
    plt.savefig("engine_max_lifetime.png")

    plt.hist(rul_dist["rul"].to_list(), bins=20, edgecolor="black")
    plt.xlabel("Min Cycles left before Failure")
    plt.ylabel("Number of Engines")
    plt.title("RUL Distribution")
    plt.savefig("rul_distribution.png")

    ## Sensor behaviour over time
    sensor_cols = [c for c in df_train_fd001.columns if "sensor measurement" in c]
    df_train_fd001_01 = df_train_fd001.sort(["unit number"]).filter(
        pl.col("unit number") == 1
    )

    ## Single engine lifecycle
    plot_single_engine_lifecycle(
        df_train_fd001_01, sensor_cols, time_col="time, in cycles"
    )
    ## Overlay multiple engines
    plot_multiple_engine_lifecycle(
        df_train_fd001, sensor_cols, time_col="time, in cycles", n_engines=7
    )
    ## Sensor vs RUL scatter
    # plot_sensor_vs_rul_scatter(
    #     df_train_fd001,
    # )
    ## Correlation Analysis
    ## Sensor to sensor correlation heatmap
    ## Sensor to RUL correlation

    ## Operational Settings
    ## Plot the 3 operational settings over time. Are they truly constant, or do they vary within a single cycle window?


if __name__ == "__main__":
    main()
