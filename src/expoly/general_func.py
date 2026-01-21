# src/expoly/general_func.py
from __future__ import annotations

import fractions
import itertools as it
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Plotly is optional; only required if you call the plotting helpers.
try:
    import plotly.express as px  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

EPS = 1e-10

__all__ = [
    "generate_extend_cube",
    "list_flatten",
    "dict_value_flatten",
    "translate_points",
    "quat2eul",
    "eul2rot_bunge",
    "eul2rot_ZYX",
    "rot2eul_XYZ",
    "quat2rot",
    "get_lcm_numerator",
    "Xori2rot",
    "rot2axis",
    "Xori2XYZ",
    "rotation_angle",
    "find_theta",
    "calculate_misorientation",
    "calculate_misorientation_and_hkl",
    "calculate_misorientation_quat",
    "calculate_misorientation_R",
    "cubic_symmetry_matrices",
    "load_x_sequence",
    "get_step_grain_ID",
    "load_alignment",
    "align_XYZ",
    "align_grain_H",
    "align_grain_FCC",
    "align_Hrange",
    "check_center_grain_general",
    "plot_area_ptm",
    "plot_area_ID",
]


# ------------------------ Basic utilities ------------------------

def generate_extend_cube(copy_size: int) -> np.ndarray:
    """
    Build a cube of offsets centered at the origin. The side length is (2*scale+1),
    where scale = (copy_size - 1) / 2, and the coordinates are scaled by 1/copy_size.
    Example: copy_size=1 -> 3x3x3; copy_size=2 -> 5x5x5.
    """
    if copy_size <= 0:
        raise ValueError("copy_size must be positive.")
    scale = (copy_size - 1) / 2
    x, y, z = np.mgrid[-scale:scale + 1, -scale:scale + 1, -scale:scale + 1]
    pos = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    return pos / copy_size


def list_flatten(lst: Iterable[Iterable]) -> list:
    """Flatten a 2D-like iterable of iterables into a single list."""
    return [e for sub in lst for e in sub]


def dict_value_flatten(d: Dict) -> list:
    """Flatten a dict's values (assumed to be iterables) into a single list."""
    return list_flatten([v for _, v in d.items()])


def translate_points(base_xyz: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Cartesian sum of an Nx3 point cloud with an Mx3 set of offsets.
    Returns an array of shape (N*M, 3).
    (This is the generalized version of your former Unit_vec.)
    """
    base_xyz = np.asarray(base_xyz, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    out = np.repeat(base_xyz, len(offsets), axis=0)
    off = np.tile(offsets, (len(base_xyz), 1))
    return out + off


# ------------------------ Orientation conversions ------------------------

def quat2eul(
    quat: Sequence[float],
    in_Ovito: bool = False,
    degrees: bool = False,
) -> np.ndarray:
    """
    Quaternion -> Bunge Euler angles (phi1, Phi, phi2), shape (3,).
    Input is assumed (w, x, y, z). If in_Ovito=True, input is (x, y, z, w).
    Returns radians by default; set degrees=True for degrees.
    """
    q = np.array(quat, dtype=float)
    if in_Ovito:
        q = np.array([q[3], q[0], q[1], q[2]], dtype=float)

    w, x, y, z = q
    at1 = np.arctan2(z, w)
    at2 = np.arctan2(x, y)

    alpha = at1 - at2 + np.pi / 2
    beta = 2 * np.arctan2(np.sqrt(x * x + y * y), np.sqrt(w * w + z * z))
    gamma = at1 + at2 + 3 * np.pi / 2

    alpha = np.mod(alpha, 2 * np.pi)
    gamma = np.mod(gamma, 2 * np.pi)

    eul = np.array([alpha, beta, gamma], dtype=float)
    return np.rad2deg(eul) if degrees else eul


def eul2rot_bunge(theta: Sequence[float], radians: bool = True) -> np.ndarray:
    """
    Bunge convention Euler angles (phi1, Phi, phi2) -> 3x3 rotation matrix.
    Matches your original formulation.
    """
    t = np.array(theta, dtype=float)
    if not radians:
        t = np.deg2rad(t)
    c0, c1, c2 = np.cos(t[0]), np.cos(t[1]), np.cos(t[2])
    s0, s1, s2 = np.sin(t[0]), np.sin(t[1]), np.sin(t[2])

    R = np.array([
        [c2 * c0 - c1 * s0 * s2,  c2 * s0 + c1 * c0 * s2,  s2 * s1],
        [-s2 * c0 - c1 * s0 * c2, -s2 * s0 + c1 * c0 * c2, c2 * s1],
        [s1 * s0,                 -s1 * c0,                c1]
    ], dtype=float)
    R[np.abs(R) < EPS] = 0.0
    return R


def eul2rot_ZYX(theta: Sequence[float], radians: bool = True) -> np.ndarray:
    """
    ZYX (yaw-pitch-roll) Euler angles -> 3x3 rotation matrix.
    Matches your original formulation.
    """
    t = np.array(theta, dtype=float)
    if not radians:
        t = np.deg2rad(t)
    c0, c1, c2 = np.cos(t[0]), np.cos(t[1]), np.cos(t[2])
    s0, s1, s2 = np.sin(t[0]), np.sin(t[1]), np.sin(t[2])

    R = np.array([
        [c0 * c1,                c0 * s1 * s2 - s0 * c2,  c0 * s1 * c2 + s0 * s2],
        [s0 * c1,                s0 * s1 * s2 + c0 * c2,  s0 * s1 * c2 - c0 * s2],
        [-s1,                    c1 * s2,                 c1 * c2]
    ], dtype=float)
    R[np.abs(R) < 1e-5] = 0.0
    return R


def rot2eul_XYZ(R: np.ndarray, degrees: bool = False) -> np.ndarray:
    """
    Approximate XYZ Euler angles from a 3x3 rotation matrix.
    Returns radians by default; set degrees=True for degrees.
    """
    R = np.asarray(R, dtype=float)
    theta = -np.arcsin(R[2, 0])
    phi = np.arctan2(R[2, 1] / np.cos(theta), R[2, 2] / np.cos(theta))
    si = np.arctan2(R[1, 0] / np.cos(theta), R[0, 0] / np.cos(theta))
    out = np.array([theta, phi, si], dtype=float)
    return np.rad2deg(out) if degrees else out


def quat2rot(quat: Sequence[float], in_Ovito: bool = False) -> np.ndarray:
    """
    Quaternion -> 3x3 rotation matrix. Input is (w, x, y, z) unless in_Ovito=True
    in which case it is (x, y, z, w). The quaternion is normalized internally.
    """
    q = np.array(quat, dtype=float)
    if in_Ovito:
        q = np.array([q[3], q[0], q[1], q[2]], dtype=float)
    n = np.linalg.norm(q)
    if n < EPS:
        raise ValueError("Invalid quaternion (near zero).")
    w, x, y, z = q / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)]
    ], dtype=float)
    return R


def get_lcm_numerator(line: np.ndarray) -> np.ndarray:
    """
    Convert a 3-vector into an integer direction by rational approximation.
    Example: [0.5, 0.25, 0] -> [2, 1, 0]
    """
    line = np.asarray(line, dtype=float)
    if np.allclose(line, np.round(line)):
        return line.astype(int)

    nums, dens = [], []
    for val in line:
        if abs(val) < EPS:
            nums.append(0)
            dens.append(1)
        else:
            f = fractions.Fraction(val).limit_denominator()
            nums.append(f.numerator)
            dens.append(f.denominator)
    nums = np.asarray(nums, dtype=np.int64)
    dens = np.asarray(dens, dtype=np.int64)

    lcm = np.lcm.reduce(dens)
    factor = (lcm // dens)
    return (factor * nums).astype(int)


def Xori2rot(X_ori: Sequence[float]) -> np.ndarray:
    """
    Build a rotation that aligns the input 3-vector X_ori with a reference
    using a ZYX Euler construction.
    """
    X_ori = np.asarray(X_ori, dtype=float)
    X_xy = np.array([X_ori[0], X_ori[1], 0.0], dtype=float)
    if np.linalg.norm(X_xy) < EPS or np.linalg.norm(X_ori) < EPS:
        return np.eye(3)
    Z = np.pi/2 - np.arccos(np.dot(X_xy, np.array([0, 1, 0])) / np.linalg.norm(X_xy))
    Y = -np.arccos(np.dot(X_ori, X_xy) / (np.linalg.norm(X_xy) * np.linalg.norm(X_ori)))
    return eul2rot_ZYX([Z, Y, 0.0], radians=True)


def rot2axis(R: np.ndarray) -> List[np.ndarray]:
    """
    Convert each column of a 3x3 rotation matrix into a small-integer axis
    (by normalizing, rounding, and applying get_lcm_numerator).
    """
    R = np.asarray(R, dtype=float)
    out = []
    for col in range(3):
        v = R[:, col]
        v = v / np.sum(np.abs(v))
        v = np.round(v, 8)
        out.append(get_lcm_numerator(v))
    return out


def Xori2XYZ(X_ori: Sequence[float]) -> List[np.ndarray]:
    """Shortcut: X_ori -> rotation -> integer axes."""
    return rot2axis(Xori2rot(X_ori))


# ------------------------ Misorientation & symmetry ------------------------

def rotation_angle(R: np.ndarray, degrees: bool = False) -> float:
    """Angle (in [0, pi]) from a rotation matrix."""
    tr = (np.trace(R) - 1.0) / 2.0
    tr = float(np.clip(tr, -1.0, 1.0))
    ang = np.arccos(tr)
    return float(np.rad2deg(ang) if degrees else ang)


def cubic_symmetry_matrices(proper_only: bool = True) -> List[np.ndarray]:
    """
    Generate cubic crystal symmetry operators.
    - proper_only=True: 24 proper rotations (det=+1)
    - proper_only=False: also include improper operations (48 total)
    """
    mats: List[np.ndarray] = []
    for perm in it.permutations(range(3)):
        P = np.zeros((3, 3), dtype=int)
        for i, j in enumerate(perm):
            P[i, j] = 1
        for signs in it.product([1, -1], repeat=3):
            S = np.diag(signs)
            M = S @ P
            det = round(np.linalg.det(M))
            if proper_only and det != 1:
                continue
            mats.append(M.astype(int))
    # deduplicate
    uniq: List[np.ndarray] = []
    seen = set()
    for M in mats:
        key = tuple(M.flatten().tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(M.astype(float))
    return uniq


def find_theta(R_ab: np.ndarray, symmetry_ops: Sequence[np.ndarray], to_hkl: bool = False):
    """
    Given a relative rotation R_ab = R_a * R_b^T, search over symmetry_ops and
    return the minimum misorientation angle. If to_hkl=True, also return the
    equivalent rotation matrix achieving that minimum.
    """
    best_ang = np.inf
    best_mat = None
    for S in symmetry_ops:
        R = S @ R_ab
        ang = rotation_angle(R, degrees=False)
        if ang < best_ang:
            best_ang = ang
            best_mat = R
    if to_hkl:
        return best_ang, best_mat  # radians, matrix
    return np.rad2deg(best_ang)


def calculate_misorientation(this_eul: Sequence[float],
                             other_eul: Sequence[float],
                             symmetry_ops: Sequence[np.ndarray]) -> float:
    """Misorientation (degrees) from two Bunge Euler triplets."""
    Ra = eul2rot_bunge(this_eul, radians=True)
    Rb = eul2rot_bunge(other_eul, radians=True)
    R = Ra @ Rb.T
    return find_theta(R, symmetry_ops, to_hkl=False)


def calculate_misorientation_and_hkl(this_eul: Sequence[float],
                                     other_eul: Sequence[float],
                                     symmetry_ops: Sequence[np.ndarray]):
    """
    Misorientation (degrees) and the corresponding (h,k,l)-like components
    derived from the minimum-angle equivalent rotation.
    """
    Ra = eul2rot_bunge(this_eul, radians=True)
    Rb = eul2rot_bunge(other_eul, radians=True)
    R = Ra @ Rb.T
    ang_rad, R_min = find_theta(R, symmetry_ops, to_hkl=True)
    s = 2 * np.sin(ang_rad)
    h = (R_min[1, 2] - R_min[2, 1]) / s
    k = (R_min[2, 0] - R_min[0, 2]) / s
    l_comp = (R_min[2, 1] - R_min[1, 2]) / s
    return np.rad2deg(ang_rad), h, k, l_comp


def calculate_misorientation_quat(this_quat: Sequence[float],
                                  other_quat: Sequence[float],
                                  symmetry_ops: Sequence[np.ndarray],
                                  in_Ovito: bool = False) -> float:
    """Misorientation (degrees) from two quaternions."""
    Ra = quat2rot(this_quat, in_Ovito=in_Ovito)
    Rb = quat2rot(other_quat, in_Ovito=in_Ovito)
    R = Ra @ Rb.T
    return find_theta(R, symmetry_ops, to_hkl=False)


def calculate_misorientation_R(Ra: np.ndarray,
                               Rb: np.ndarray,
                               symmetry_ops: Sequence[np.ndarray]) -> float:
    """Misorientation (degrees) from two rotation matrices."""
    R = Ra @ Rb.T
    return find_theta(R, symmetry_ops, to_hkl=False)


# Default 24 proper cubic symmetry operations
SYMMETRY_CUBIC_24 = cubic_symmetry_matrices(proper_only=True)


# ------------------------ Alignment / frame mapping (no hardcoded paths) ------------------------

def load_x_sequence(path: Path | str, sep: str = "\t") -> pd.DataFrame:
    """
    Load frame-to-frame grain ID mapping (columns like A1, A2, ...).
    """
    p = Path(path)
    return pd.read_csv(p, sep=sep)


def get_step_grain_ID(grain_ID: int, in_frame: int, target_frame: int,
                      x_sequence: pd.DataFrame, return_all: bool = False):
    """
    Map a grain ID between frames using an X-sequence dataframe.
    Provide the dataframe via load_x_sequence() and pass it here.
    """
    a = f"A{int(in_frame)}"
    b = f"A{int(target_frame)}"
    subset = x_sequence[x_sequence[a] == grain_ID]
    if return_all:
        return subset
    if subset.empty:
        return 0
    return int(subset[b].iloc[0])


def load_alignment(path: Path | str) -> pd.DataFrame:
    """
    Load cumulative alignment shifts per frame; columns: HZ, HY, HX.
    """
    p = Path(path)
    df = pd.read_csv(p, header=None, names=['HZ', 'HY', 'HX'])
    return df


def _sum_alignment(alignment: pd.DataFrame, start: int, end: int) -> np.ndarray:
    """Sum alignment deltas from start to end (exclusive) with direction handling."""
    if start == end:
        return np.zeros(3, dtype=float)
    lo, hi = sorted((start, end))
    v = alignment.iloc[lo:hi].sum().to_numpy(dtype=float)
    return v if start < end else -v


def align_XYZ(XYZ: Sequence[float], from_frame: int, to_frame: int,
              alignment: pd.DataFrame) -> List[float]:
    """
    Apply alignment shifts to a single coordinate [X,Y,Z] from one frame to another.
    """
    XYZ = np.asarray(XYZ, dtype=float)
    delta = _sum_alignment(alignment, from_frame, to_frame)
    # Map (HZ,HY,HX) accumulated shifts to (X,Y,Z) order
    out = XYZ + np.array([delta[2], delta[1], delta[0]], dtype=float)
    return out.tolist()


def align_grain_H(Grain_: pd.DataFrame, from_frame: int, to_frame: int,
                  alignment: pd.DataFrame) -> pd.DataFrame:
    """
    Apply alignment to a dataframe with columns ['HZ','HY','HX'].
    Returns a new dataframe.
    """
    delta = _sum_alignment(alignment, from_frame, to_frame)
    out = Grain_.copy()
    out['HZ'] += delta[0]
    out['HY'] += delta[1]
    out['HX'] += delta[2]
    return out


def align_grain_FCC(Grain_: pd.DataFrame, from_frame: int, to_frame: int,
                    alignment: pd.DataFrame) -> pd.DataFrame:
    """
    Apply alignment to a dataframe with columns ['X','Y','Z'].
    Returns a new dataframe.
    """
    delta = _sum_alignment(alignment, from_frame, to_frame)
    out = Grain_.copy()
    out['Z'] += delta[0]
    out['Y'] += delta[1]
    out['X'] += delta[2]
    return out


def align_Hrange(HX_range: Sequence[int], HY_range: Sequence[int], HZ_range: Sequence[int],
                 from_frame: int, to_frame: int,
                 alignment: pd.DataFrame) -> Tuple[List[float], List[float], List[float]]:
    """
    Apply alignment to a bounding box expressed as HX/HY/HZ ranges.
    Returns three [low, high] lists for HX/HY/HZ after alignment.
    """
    low_xyz = [HX_range[0], HY_range[0], HZ_range[0]]
    high_xyz = [HX_range[1], HY_range[1], HZ_range[1]]
    post_low = align_XYZ(low_xyz, from_frame, to_frame, alignment)
    post_high = align_XYZ(high_xyz, from_frame, to_frame, alignment)
    return [post_low[0], post_high[0]], [post_low[1], post_high[1]], [post_low[2], post_high[2]]


# ------------------------ Selection / visualization (Plotly optional) ------------------------

def check_center_grain_general(DATA_: pd.DataFrame, buffer: float, min_size: int) -> List[int]:
    """
    Select grain IDs that are sufficiently away from the boundaries and
    have at least `min_size` points. Requires columns: X, Y, Z, grain-ID.
    """
    ids = []
    uniq = np.unique(DATA_['grain-ID'])
    xmin, xmax = DATA_['X'].min(), DATA_['X'].max()
    ymin, ymax = DATA_['Y'].min(), DATA_['Y'].max()
    zmin, zmax = DATA_['Z'].min(), DATA_['Z'].max()

    for gid in uniq:
        g = DATA_[DATA_['grain-ID'] == gid]
        if (len(g) > min_size and
            g['X'].min() > xmin + buffer and g['X'].max() < xmax - buffer and
            g['Y'].min() > ymin + buffer and g['Y'].max() < ymax - buffer and
            g['Z'].min() > zmin + buffer and g['Z'].max() < zmax - buffer):
            ids.append(int(gid))
    return ids


def _ensure_plotly():
    if not _HAS_PLOTLY:
        raise ImportError("Plotly is not installed. Install with `pip install 'expoly[viz]'`.")


def plot_area_ptm(DATA_: pd.DataFrame, Xmax, Xmin, Ymax, Ymin, Zmax, Zmin):
    """
    Build a 3D scatter plot for a selected region colored by 'ptm-type'.
    Returns a Plotly figure; the caller decides whether to `fig.show()`.
    """
    _ensure_plotly()
    data = DATA_[(DATA_['X'] > Xmin) & (DATA_['X'] < Xmax) &
                 (DATA_['Y'] > Ymin) & (DATA_['Y'] < Ymax) &
                 (DATA_['Z'] > Zmin) & (DATA_['Z'] < Zmax)].copy()
    if 'ptm-type' not in data.columns and 'ptm_type' in data.columns:
        data['ptm-type'] = data['ptm_type'].astype('string')
    else:
        data['ptm-type'] = data['ptm-type'].astype('string')

    fig = px.scatter_3d(
        data, x='X', y='Y', z='Z', color='ptm-type',
        color_discrete_map={
            '0.0': 'rgba(5, 5, 5,0.1)',
            '1.0': 'rgba(240,240,240,0.0)',
            '2.0': 'rgba(63,158,22,1)',
            '3.0': 'rgba(247,192,38,0.6)',
            '4.0': 'rgba(209,82,78,0.6)',
        }
    )
    return fig


def plot_area_ID(DATA_: pd.DataFrame, Xmax, Xmin, Ymax, Ymin, Zmax, Zmin):
    """
    Build a 3D scatter plot for a selected region colored by 'grain-ID'.
    Returns a Plotly figure; the caller decides whether to `fig.show()`.
    """
    _ensure_plotly()
    data = DATA_[(DATA_['X'] > Xmin) & (DATA_['X'] < Xmax) &
                 (DATA_['Y'] > Ymin) & (DATA_['Y'] < Ymax) &
                 (DATA_['Z'] > Zmin) & (DATA_['Z'] < Zmax)]
    fig = px.scatter_3d(data, x='X', y='Y', z='Z', color='grain-ID')
    return fig
