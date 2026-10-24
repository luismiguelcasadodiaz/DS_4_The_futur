import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    """Read a CSV file and display a heatmap of its correlation matrix.

    The "knight" column is encoded as numeric (Jedi -> 1, Sith -> 0) so it
    can be included in the correlation. The correlation matrix is printed
    to stdout, then plotted as a heatmap with matplotlib/seaborn.

    Args:
        path: Path to the CSV file. It must contain a "knight" column with
            values "Jedi" or "Sith"; all other columns should be numeric.

    Returns:
        None. The correlation matrix is printed and the heatmap is shown
        in a window.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
        KeyError: If the CSV has no "knight" column.
    """
    df = pd.read_csv(path)
    if "knight" not in df.columns:
        raise KeyError("Column 'knight' not found in the DataFrame")
    df["knight"] = df["knight"].map({"Jedi": 1, "Sith": 0})
    print(df.corr())
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Heatmap of Correlation Matrix")
    plt.subplots_adjust(left=0.192,
                        bottom=0.254,
                        right=0.99,
                        top=0.927,
                        wspace=0.2,
                        hspace=0.2)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(path_test(sys.argv[1]))
    else:
        print("python ./Heatmap.py <true file's path>")
        sys.exit(1)
