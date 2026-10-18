import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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


def main(truth: str, predict: str) -> None:
    df_t = pd.read_csv(truth, header=None)
    df_p = pd.read_csv(predict, header=None)
    assert df_t.shape[0] == df_p.shape[0], "Truth & predict len doe not match"
    class_t = sorted(df_t[0].unique())
    class_p = sorted(df_p[0].unique())
    assert class_t == class_p, "Different classes in Truth & predict"
    df_cm = pd.DataFrame(0, index=class_t, columns=class_p)
    for t, p in zip(df_t[0].to_list(), df_p[0].to_list()):
        df_cm.loc[t, p] += 1
    print(df_cm)



if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(path_test(sys.argv[1]), path_test(sys.argv[2]))
    else:
        print("python ./Confusion_matrix.py <true file's path> <Predict file's path>")
        sys.exit(1)
