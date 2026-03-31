import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


def _to_2d_array(a):
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array with shape (n_points, n_objectives).")
    if arr.shape[0] == 0:
        raise ValueError("Pareto frontier is empty.")
    return arr


def _to_minimization(frontier, maximize_mask=None):
    f = frontier.copy()
    if maximize_mask is None:
        return f

    mask = np.asarray(maximize_mask, dtype=bool)
    if mask.shape[0] != f.shape[1]:
        raise ValueError("maximize_mask length must match number of objectives.")
    f[:, mask] = -f[:, mask]
    return f


def _normalize(frontier, reference_front=None):
    if reference_front is None:
        combined = frontier
    else:
        combined = np.vstack([frontier, reference_front])

    mins = np.min(combined, axis=0)
    maxs = np.max(combined, axis=0)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))

    frontier_norm = (frontier - mins) / denom
    if reference_front is None:
        return frontier_norm, None
    reference_norm = (reference_front - mins) / denom
    return frontier_norm, reference_norm


def _spread_mean_pairwise(frontier):
    n = len(frontier)
    if n < 2:
        return np.nan
    d = np.sqrt(((frontier[:, None, :] - frontier[None, :, :]) ** 2).sum(axis=2))
    i, j = np.triu_indices(n, k=1)
    return float(np.mean(d[i, j]))


def calculate_pareto_metrics(
    frontier,
    reference_front=None,
    ref_point=None,
    maximize_mask=None,
    normalize=True,
):
    """
    Compute HV, spread, IGD, and mean distance to heaven.

    Parameters
    ----------
    frontier : array-like, shape (n_points, n_objectives)
        Pareto frontier values.
    reference_front : array-like, optional
        True/reference Pareto front for IGD.
    ref_point : array-like, optional
        Hypervolume reference point in minimization space.
        If omitted, uses [1.1, ..., 1.1] on normalized objectives.
    maximize_mask : array-like of bool, optional
        True for objectives that are maximize in the input. These are sign-flipped.
    normalize : bool, default=True
        If True, metrics are computed on normalized objectives.

    Returns
    -------
    dict
        Keys: hv, spread, igd, mdh (mean distance to heaven)
    """
    f = _to_2d_array(frontier)
    ref = None if reference_front is None else _to_2d_array(reference_front)

    f = _to_minimization(f, maximize_mask=maximize_mask)
    if ref is not None:
        ref = _to_minimization(ref, maximize_mask=maximize_mask)

    if normalize:
        f_eval, ref_eval = _normalize(f, ref)
    else:
        f_eval, ref_eval = f, ref

    if ref_point is None:
        hv_ref = np.ones(f_eval.shape[1], dtype=float) * 1.1
    else:
        hv_ref = np.asarray(ref_point, dtype=float)

    hv = float(HV(ref_point=hv_ref)(f_eval))
    spread = _spread_mean_pairwise(f_eval)
    igd = float(IGD(ref_eval)(f_eval)) if ref_eval is not None else np.nan

    heaven = np.zeros(f_eval.shape[1], dtype=float)
    mdh = float(np.mean(np.linalg.norm(f_eval - heaven, axis=1)))

    return {"hv": hv, "spread": spread, "igd": igd, "mdh": mdh}

