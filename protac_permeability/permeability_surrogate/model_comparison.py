from typing import Callable, Tuple

import numpy as np
from scipy.stats import t as students_t
from sklearn.metrics import r2_score

N_REPLICATIONS = 5
DEGREES_OF_FREEDOM = 5


def default_two_fold_split(
    X: np.ndarray, y: np.ndarray, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly shuffle and split indices into two equal-sized halves.

    Args:
        X: Feature matrix; only its length is used.
        y: Target array; only its length is used.
        random_state: Seed controlling the shuffle.

    Returns:
        A tuple (idx1, idx2) of disjoint index arrays covering all of y,
        each holding half (+/- 1 row) of the data.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(y))
    half = len(idx) // 2
    return idx[:half], idx[half:]


def five_by_two_cv_paired_ttest(
    fit_predict_a: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    fit_predict_b: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = r2_score,
    splitter: Callable[
        [np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray]
    ] = default_two_fold_split,
    min_fold_fraction: float = 0.4,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Compare two learning algorithms with Dietterich's 5x2cv paired t test.

    Implements section 3.5 of Dietterich (1998), "Approximate Statistical
    Tests for Comparing Supervised Classification Learning Algorithms"
    (https://web.engr.oregonstate.edu/~tgd/publications/dietterich-approximate-statistical-tests-nc1998.pdf),
    generalized from classification error rate to an arbitrary paired
    regression metric. Runs 5 replications of 2-fold cross-validation: in
    each replication both algorithms are trained on one half and tested
    on the other, and vice versa, giving two paired metric differences
    per replication. The test statistic's numerator is only the very
    first replication's difference (divided by the pooled variance across
    all 5 replications), not the mean of all 10 differences, since the
    paper found the mean to be badly overestimated due to correlation
    between the two folds of a replication.

    `metric` is assumed to be "higher is better" (like the default R2);
    a positive returned statistic means `fit_predict_a` outperforms
    `fit_predict_b`. Pass a metric that negates a "lower is better" score
    (e.g. `lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)`) to
    compare on those terms instead.

    This test relies on several independence assumptions (detailed in the
    paper) that are known to be only approximately true in practice, so
    treat it as a heuristic rather than an exact test.

    Args:
        fit_predict_a: Callable(train_X, train_y, test_X) -> predictions,
            for algorithm A.
        fit_predict_b: Same signature, for algorithm B.
        X: Full feature matrix to split into folds.
        y: Full target array, aligned with X.
        metric: Callable(y_true, y_pred) -> float, higher-is-better.
            Defaults to R2.
        splitter: Callable(X, y, random_state) -> (idx1, idx2) producing
            one 2-fold split. Defaults to a plain random 50/50 shuffle
            split; called once per replication with a different seed.
        min_fold_fraction: Minimum fraction of the data each of the two
            folds returned by `splitter` must contain. If `splitter`
            returns a more skewed split than this, that replication
            silently falls back to `default_two_fold_split` instead,
            since the test statistic assumes roughly equal-sized folds.
        random_state: Base seed; replication i uses seed
            `random_state + i`.

    Returns:
        A tuple (t_statistic, p_value) for the two-sided test, with the
        5 degrees of freedom fixed by the procedure (not derived from the
        data size). If both algorithms agree exactly on every replication
        (zero pooled variance), returns (0.0, 1.0) when they also tie on
        the first replication, or (+/-inf, 0.0) otherwise.
    """
    first_replication_diff = None
    variances = []

    for i in range(N_REPLICATIONS):
        seed = random_state + i
        idx1, idx2 = splitter(X, y, seed)
        fold_fraction = len(idx1) / len(y)
        if fold_fraction < min_fold_fraction or fold_fraction > 1 - min_fold_fraction:
            idx1, idx2 = default_two_fold_split(X, y, seed)

        X1, y1 = X[idx1], y[idx1]
        X2, y2 = X[idx2], y[idx2]

        # Train on half 1, test on half 2.
        diff_1 = metric(y2, fit_predict_a(X1, y1, X2)) - metric(
            y2, fit_predict_b(X1, y1, X2)
        )
        # Train on half 2, test on half 1.
        diff_2 = metric(y1, fit_predict_a(X2, y2, X1)) - metric(
            y1, fit_predict_b(X2, y2, X1)
        )

        mean_diff = (diff_1 + diff_2) / 2
        variances.append((diff_1 - mean_diff) ** 2 + (diff_2 - mean_diff) ** 2)

        if i == 0:
            # t~'s numerator uses only the first replication's paired
            # difference; see docstring for why the 10-difference mean
            # is not used instead.
            first_replication_diff = diff_1

    pooled_variance = np.mean(variances)
    if pooled_variance == 0:
        # Deterministic algorithms agreeing exactly on every replication:
        # zero variance is a genuine (not numerically noisy) result, so
        # handle it directly instead of dividing by zero.
        if first_replication_diff == 0:
            return 0.0, 1.0
        return np.sign(first_replication_diff) * np.inf, 0.0

    t_statistic = first_replication_diff / np.sqrt(pooled_variance)
    p_value = 2 * students_t.sf(np.abs(t_statistic), DEGREES_OF_FREEDOM)
    return t_statistic, p_value
