import sys
import os
import pandas as pd


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
    """
    Compute accuracy from a 2x2 confusion matrix.

    Accuracy is the proportion of all predictions that are correct,
    counting both positive and negative classes.

    Parameters
    ----------
    df_cm : pd.DataFrame
        2x2 confusion matrix with actual values as rows and predicted
        values as columns, with the positive class second::

                          Pred. Negative   Pred. Positive
            Act. Negative       TN               FP
            Act. Positive       FN               TP

    Returns
    -------
    float
        Accuracy between 0 and 1. Returns 0.0 if the confusion
        matrix is empty (total = 0).

    See Also
    --------
    precision : Compute precision, TP / (TP + FP).
    recall : Compute recall, TP / (TP + FN).
    specificity : Compute specificity, TN / (TN + FP).
    F1score : Harmonic mean of precision and recall.

    Notes
    -----
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Accuracy can be misleading on imbalanced datasets: if 95% of
    samples are negative, a model that always predicts negative
    scores 0.95 while never detecting a positive. In that case,
    prefer precision, recall or F1 score.

    The positive class must be in row 1 and column 1, matching the
    layout of scikit-learn's ``confusion_matrix`` for labels [0, 1].
    """
    TN = df_cm.iloc[0, 0]
    TP = df_cm.iloc[1, 1]
    total = df_cm.values.sum()
    num = (TP + TN) * 1.0
    den = total
    return num / den if den != 0 else 0.0


def precision(df_cm: pd.Dataframe, catego=1) -> float:
    """
    Compute precision from a 2x2 confusion matrix.

    Precision is the proportion of positive predictions that are
    actually correct: of everything the model labelled positive,
    how much really was positive?

    Parameters
    ----------
    df_cm : pd.DataFrame
        2x2 confusion matrix with actual values as rows and predicted
        values as columns, with the positive class second::

                          Pred. Negative   Pred. Positive
            Act. Negative       TN               FP
            Act. Positive       FN               TP

    Returns
    -------
    float
        Precision between 0 and 1. Returns 0.0 if the model made no
        positive predictions (TP + FP = 0).

    See Also
    --------
    recall : Compute recall, TP / (TP + FN).
    F1score : Harmonic mean of precision and recall.

    Notes
    -----
    Precision = TP / (TP + FP)

    The positive class must be in row 1 and column 1, matching the
    layout of scikit-learn's ``confusion_matrix`` for labels [0, 1].
    """
    num = df_cm.iloc[catego, catego] * 1.0
    den = df_cm.iloc[:, catego].sum()
    return num / den if den != 0 else 0.0


def recall(df_cm: pd.Dataframe, catego=1) -> float:
    """
    Compute recall from a 2x2 confusion matrix.

    Recall (also called sensitivity or true positive rate) is the
    proportion of actual positives the model correctly identified:
    of all the cases that really were positive, how many did the
    model catch?

    Parameters
    ----------
    df_cm : pd.DataFrame
        2x2 confusion matrix with actual values as rows and predicted
        values as columns, with the positive class second::

                          Pred. Negative   Pred. Positive
            Act. Negative       TN               FP
            Act. Positive       FN               TP

    Returns
    -------
    float
        Recall between 0 and 1. Returns 0.0 if there are no actual
        positives in the data (TP + FN = 0).

    See Also
    --------
    precision : Compute precision, TP / (TP + FP).
    F1score : Harmonic mean of precision and recall.

    Notes
    -----
    Recall = TP / (TP + FN)

    The positive class must be in row 1 and column 1, matching the
    layout of scikit-learn's ``confusion_matrix`` for labels [0, 1].
    """
    num = df_cm.iloc[catego, catego] * 1.0
    den = df_cm.iloc[catego].sum()
    return num / den if den != 0 else 0.0


def specifity(df_cm: pd.Dataframe) -> float:
    """
    Compute specificity from a 2x2 confusion matrix.

    Specificity (also called true negative rate) is the proportion
    of actual negatives the model correctly identified: of all the
    cases that really were negative, how many did the model label
    as negative?

    Parameters
    ----------
    df_cm : pd.DataFrame
        2x2 confusion matrix with actual values as rows and predicted
        values as columns, with the positive class second::

                          Pred. Negative   Pred. Positive
            Act. Negative       TN               FP
            Act. Positive       FN               TP

    Returns
    -------
    float
        Specificity between 0 and 1. Returns 0.0 if there are no
        actual negatives in the data (TN + FP = 0).

    See Also
    --------
    recall : Compute recall (sensitivity), TP / (TP + FN).
    precision : Compute precision, TP / (TP + FP).

    Notes
    -----
    Specificity = TN / (TN + FP)

    Specificity is the counterpart of recall for the negative class.
    The positive class must be in row 1 and column 1, matching the
    layout of scikit-learn's ``confusion_matrix`` for labels [0, 1].
    """
    TN = df_cm.iloc[0, 0]
    FP = df_cm.iloc[0, 1]
    num = TN * 1.0
    den = TN + FP
    return num / den if den != 0 else 0.0


def F1score(df_cm: pd.Dataframe, catego=1) -> float:
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
    p = precision(df_cm, catego)
    r = recall(df_cm, catego)
    num = 2.0 * p * r
    den = p + r
    return num / den if den != 0 else 0.0


def get_class(df_cm: pd.DataFrame, num: int) -> str:
    """
    Get the class label from a confusion matrix DataFrame.

    Parameters
    ----------
    df_cm : pd.DataFrame
        Confusion matrix with actual values as rows and predicted
        values as columns.
    num : int
        Index of the class to retrieve
        (0 for the first class, 1 for the second).

    Returns
    -------
    str
        The class label corresponding to the specified index.
    """
    return str(df_cm.index[num])


def get_total(df_cm: pd.DataFrame, num: int) -> int:
    """
    Get the total number of instances for a specific class
    from a confusion matrix.

    Parameters
    ----------
    df_cm : pd.DataFrame
        Confusion matrix with actual values as rows and predicted
        values as columns.
    num : int
        Index of the class to retrieve the total for
        (0 for the first class, 1 for the second).

    Returns
    -------
    int
        The total number of instances for the specified class.
    """
    return df_cm.iloc[num].sum()


def print_metrics_table(df_cm: pd.DataFrame) -> None:
    """
    Print a per-class classification report from a confusion matrix.

    For each class, prints precision, recall, F1 score and the total
    number of actual samples (support), followed by the overall
    accuracy and the confusion matrix itself. The layout is similar
    to scikit-learn's ``classification_report``.

    Parameters
    ----------
    df_cm : pd.DataFrame
        Square confusion matrix with actual values as rows and
        predicted values as columns. The index and columns hold the
        class labels, in the same order. Works for binary and
        multi-class problems.

    Returns
    -------
    None
        The report is printed to standard output.

    See Also
    --------
    precision : Per-class precision, TP / (TP + FP).
    recall : Per-class recall, TP / (TP + FN).
    F1score : Per-class harmonic mean of precision and recall.
    accuracy : Overall proportion of correct predictions.

    Notes
    -----
    Each function in the report is called as ``func(df_cm, idx)``,
    where ``idx`` is the position of the class in ``df_cm.index``.
    The "Total" column is the number of actual samples of each class
    (the row sum), and the accuracy row shows the total number of
    samples. The first column adapts to the longest class label;
    numeric columns have a minimum width of ``column_width``
    (10 characters).
    """
    column_width = 10
    extra_text = "Accuracy"
    classes_length = max(
        len(str(cls)) for cls in df_cm.index.append(pd.Index([extra_text])))
    functions = [get_class, precision, recall, F1score, get_total]
    headers = ["Classes", "Precision", "Recall", "F1-score", "Total"]
    head_w = [max(len(headers[0]), classes_length)]\
        + [max(len(header), column_width) for header in headers[1:]]

    headers_formats = ["", ".2f", ".2f", ".2f", ""]
    print("-" * sum(head_w))
    for title, width in zip(headers, head_w):
        print(f"{title:>{width}}", end="")
    print("")
    for idx in range(len(df_cm.index)):
        for func, width, fmt in zip(functions, head_w, headers_formats):
            print(f"{func(df_cm, idx):>{width}{fmt}}", end="")
        print("")

    print("")
    empty = ""
    print(f"{extra_text:>{head_w[0]}}{empty:>{head_w[1]}}", end="")
    print(f"{empty:>{head_w[2]}}{accuracy(df_cm):>{head_w[3]}.2f}", end="")
    print(f"{df_cm.values.sum():>{head_w[4]}}")
    print("-" * sum(head_w))
    print(df_cm)
    print("-" * sum(head_w))


def main(truth: str, predict: str) -> None:
    """
    Build a confusion matrix from two CSV files and print its metrics.

    Reads the true labels and the predicted labels, counts every
    (actual, predicted) pair into a confusion matrix, and prints a
    per-class report with precision, recall, F1 score, total and
    overall accuracy.

    Parameters
    ----------
    truth : str
        Path to a CSV file with the true labels, one per row, in the
        first column and without a header.
    predict : str
        Path to a CSV file with the predicted labels, in the same
        format and row order as ``truth``.

    Returns
    -------
    None
        The report is printed to standard output.

    Raises
    ------
    AssertionError
        If the two files have a different number of rows, or if they
        do not contain exactly the same set of classes.

    See Also
    --------
    print_metrics_table : Print the per-class report.

    Notes
    -----
    Row ``i`` of ``predict`` is compared with row ``i`` of ``truth``.
    The confusion matrix has actual classes as rows and predicted
    classes as columns, both sorted in ascending order.
    """
    df_t = pd.read_csv(truth, header=None)
    df_p = pd.read_csv(predict, header=None)
    assert df_t.shape[0] == df_p.shape[0], "Truth & predict len does not match"

    # In case the model did not predict all classes,
    # we need to add them to the confusion matrix
    classes = sorted(set(df_t[0]) | set(df_p[0]))

    df_cm = pd.DataFrame(0, index=classes, columns=classes)
    for t, p in zip(df_t[0].to_list(), df_p[0].to_list()):
        df_cm.loc[t, p] += 1
    print_metrics_table(df_cm)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(path_test(sys.argv[1]), path_test(sys.argv[2]))
    else:
        print("python ./Confusion_matrix.py \
            <true file's path> <Predict file's path>")
        sys.exit(1)
