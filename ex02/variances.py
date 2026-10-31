import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler


carpeta_script = Path(__file__).resolve().parent


def path_test(path: str) -> str:
    """Validate that a given path points to a readable CSV/TXT/JSON file.

    Resolves the provided path to an absolute path and runs a series of
    checks to ensure the file exists, is a regular file (not a directory),
    is readable by the current user, and has a .csv, .txt, or .json extension.

    Args:
        path: A relative or absolute filesystem path to validate.

    Returns:
        The resolved absolute path to the validated CSV, TXT, or JSON file.

    Raises:
        AssertionError: If the path does not exist, is not a regular file,
            is not readable, or does not have a .csv/.txt/.json extension.
    """
    abspath = os.path.abspath(path)
    assert os.path.exists(abspath), f"Wrong Path {path}"
    assert os.path.isfile(abspath), f"{path} is not a file"
    assert os.access(abspath, os.R_OK), f"User can not read permit on {path}"
    _, ext = os.path.splitext(abspath)
    assert ext.lower() in (".csv", ".json", ".txt"), \
        f"Expected a CSV file, got '{ext[1:]}'"
    return abspath


def main(path: str) -> None:

    df = pd.read_csv(path)
    if "knight" not in df.columns:
        raise KeyError("Column 'knight' not found in the DataFrame")

    # Drop the 'knight' column as it is not needed for variance analysis
    df = df.drop(columns=["knight"])

    df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # WIth StandarScaler, all variable have the same variance,
    # so it is not useful to see the variance of each variable.
    # scaler = StandardScaler()
    # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    variances = df.var().sort_values(ascending=False)
    totvar = variances.sum()
    variances_percentage = (variances / totvar) * 100
    accumulated_variances = variances_percentage.cumsum()

    # Nº de variables necesarias para llegar al 90 %
    threshold = 90
    n_vars = int((accumulated_variances < threshold).sum()) + 1

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # set up secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Variance", color="blue")
    ax2.bar(variances.index, variances.values,
            color="blue", alpha=0.6,
            label="Variance per feature")

    # set up primary y-axis
    ax1.plot(accumulated_variances.index, accumulated_variances.values,
             color="red", marker="o",
             label="Accumulated Variance")
    ax1.set_ylabel("Accumulated Variance (%)", color="red")
    ax1.set_xlabel("Features")
    ax1.set_ylim(0, 105)
    ax1.set_title("Variance and Accumulated Variance per Feature")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_xticks(range(len(variances.index)))
    ax1.set_xticklabels(variances.index, rotation=45, ha="right")

    # add horizontal line at 90% accumulated variance
    ax1.axhline(y=threshold, color="green",
                linestyle="--",
                label=f"{threshold}% Threshold")
    ax1.text(n_vars, threshold - 4,
             f"{threshold}% Threshold", color="green",
             fontsize=10, va="bottom")

    # add vertical line at the number of variables needed to reach 90%
    ax1.axvline(x=n_vars - 1,
                color="purple", linestyle="--",
                label=f"{n_vars} Variables Needed")
    ax1.text(n_vars, threshold - 8,
             f"{n_vars} Variables explained\
    {accumulated_variances.iloc[n_vars - 1]:.2f}%",
             color="purple", fontsize=10)
    plt.subplots_adjust(left=0.05, right=0.927, top=0.948, bottom=0.2)
    plt.savefig(carpeta_script / "variances_plot.png",
                dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(path_test(sys.argv[1]))
    else:
        print("python ./variances.py <true file's path>")
        sys.exit(1)
