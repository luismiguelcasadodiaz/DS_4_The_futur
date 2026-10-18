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

def accuracy(df_cm: pd.Dataframe) -> float:
    pass
    return 0.0

def precision(df_cm: pd.Dataframe) -> float:
    pass
    return 0.0

def recall(df_cm: pd.Dataframe) -> float:
    pass
    return 0.0

def specifity(df_cm: pd.Dataframe) -> float:
    pass
    return 0.0    

def F1score(df_cm: pd.Dataframe) -> float:
    """
    Compute the F1 score from a confusion matrix.

    The F1 score is the harmonic mean of precision and recall,
    balancing false positives and false negatives in a single metric.

    Parameters
    ----------
    df_cm : pd.DataFrame
        Confusion matrix with actual values as rows and predicted
        values as columns.

    Returns
    -------
    float
        F1 score between 0 and 1, where 1 means perfect precision
        and recall. Returns 0.0 if both precision and recall are 0.

    See Also
    --------
    precision : Compute precision, TP / (TP + FP).
    recall : Compute recall, TP / (TP + FN).

    Notes
    -----
    F1 = 2 * (precision * recall) / (precision + recall)
    When precision + recall is 0, the F1 score is undefined. In that
    case the function returns 0.0 instead of raising ZeroDivisionError,
    matching scikit-learn's default behaviour.    
    """    
    p = precision(df_cm)
    r = recall df_cm
    num = 2.0 * p * r
    den = p + r
    return num / den if den != 0 else 0.0  


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
