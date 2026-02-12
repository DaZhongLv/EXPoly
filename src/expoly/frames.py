# src/expoly/frames.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py as hdf
import numpy as np
import pandas as pd

# ----------------- Common HDF5 helpers -----------------


def find_dataset_keys(
    f: hdf.File, target_name: str, prefer_groups: Optional[List[str]] = None
) -> List[str]:
    """
    Locate a dataset named `target_name` (case-insensitive) in the HDF5 file `f`.
    Returns the sequence of keys from the root to the dataset, e.g.
    ["DataContainers","ImageDataContainer","CellData","FeatureIds"].
    If multiple matches exist, prefer one whose path contains an element from
    `prefer_groups` (earlier items in `prefer_groups` have higher priority).
    """
    hits: List[List[str]] = []

    def visit(name, obj):
        if isinstance(obj, hdf.Dataset):
            if name.split("/")[-1].lower() == target_name.lower():
                keys = [k for k in name.split("/") if k]
                hits.append(keys)

    f.visititems(visit)
    if not hits:
        raise KeyError(f"Dataset named '{target_name}' not found.")

    def score(keys: List[str]) -> Tuple[int, int]:
        if prefer_groups:
            rank = len(prefer_groups)
            for i, g in enumerate(prefer_groups):
                if any(g in k for k in keys):
                    rank = i
                    break
        else:
            rank = 0
        return (rank, len(keys))

    hits.sort(key=score)
    return hits[0]


def assign_fields(
    f: hdf.File, self_obj, mapping: Dict[str, str], prefer_groups: Optional[List[str]] = None
) -> None:
    """
    Read datasets according to a mapping {attribute_name: dataset_basename}
    and assign the array to the corresponding attribute on `self_obj`.
    """
    for attr, ds_basename in mapping.items():
        keys = find_dataset_keys(f, ds_basename, prefer_groups=prefer_groups)
        obj = f
        for k in keys:
            obj = obj[k]
        setattr(self_obj, attr, obj[()])


# ----------------- Small utilities: decouple from general_func -----------------


def _unit_vec(base_xyz: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Equivalent to your legacy `general_func.Unit_vec`:
    Given N×3 base points and M×3 offsets, return the Cartesian sum of shape (N*M)×3.
    """
    base_xyz = np.asarray(base_xyz, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    out = np.repeat(base_xyz, len(offsets), axis=0)
    off = np.tile(offsets, (len(base_xyz), 1))
    return out + off


def _safe_voxel_id(
    grain_id_arr: np.ndarray, z: np.ndarray, y: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """
    Safely fetch voxel grain IDs, supporting both 3D and 4D arrays.
    If 4D, the last axis is assumed to be size-1 and index 0 is used.
    """
    if grain_id_arr.ndim == 3:
        return grain_id_arr[z, y, x]
    elif grain_id_arr.ndim == 4:
        return grain_id_arr[z, y, x, 0]
    else:
        raise ValueError(f"Unexpected GrainId ndim={grain_id_arr.ndim}")


# ----------------- Frame object -----------------


@dataclass
class Frame:
    path: str | Path
    prefer_groups: Optional[List[str]] = None
    mapping: Optional[Dict[str, str]] = None

    # Loaded fields
    GrainId: np.ndarray = None
    Euler: np.ndarray = None
    Num_NN: np.ndarray = None
    Num_list: np.ndarray = None
    Dimension: np.ndarray = None

    # Derived fields
    fid: np.ndarray = None
    eul: np.ndarray = None
    _feature_index_map: Dict[int, int] = None  # real grain id -> 0-based contiguous index

    def __post_init__(self):
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(p)

        default_mapping = {
            "GrainId": "FeatureIds",
            "Euler": "EulerAngles",
            "Num_NN": "NumNeighbors",
            "Num_list": "NeighborList",
            "Dimension": "DIMENSIONS",
            # Optional: "IPF":"IPFColor", "CI":"Confidence Index"
        }
        mapping = self.mapping or default_mapping
        prefer = self.prefer_groups or ["CellData", "CellFeatureData", "_SIMPL_GEOMETRY"]

        with hdf.File(p, "r") as f:
            assign_fields(f, self, mapping, prefer_groups=prefer)

        # Normalize shapes and types
        self.GrainId = np.asarray(self.GrainId)
        self.Euler = np.asarray(self.Euler)
        self.Num_NN = np.asarray(self.Num_NN).reshape(-1)  # -> (N,)
        self.Num_list = np.asarray(self.Num_list)
        self.Dimension = np.asarray(self.Dimension).reshape(-1)

        self.fid = self.GrainId.flatten()
        self.eul = self.Euler.reshape(-1, 3)

        uniq_ids = np.unique(self.fid)
        self._feature_index_map = {int(fid): i for i, fid in enumerate(uniq_ids)}

    # ---------- Core dimensions (SIMPL typically stores [X, Y, Z]) ----------
    @property
    def HX_lim(self) -> int:  # X
        return int(self.Dimension[0])

    @property
    def HY_lim(self) -> int:  # Y
        return int(self.Dimension[1])

    @property
    def HZ_lim(self) -> int:  # Z
        return int(self.Dimension[2])

    # ---------- Neighbor list slicing ----------
    @property
    def _neighbor_starts(self) -> np.ndarray:
        """
        For each feature i, the neighbor slice is [starts[i] : starts[i] + Num_NN[i]].
        Precompute starts via cumsum to avoid per-call loops.
        """
        nn = self.Num_NN.astype(np.int64)
        starts = np.empty_like(nn)
        np.cumsum(np.r_[0, nn[:-1]], out=starts)
        return starts

    def _index_from_id(self, grain_id: int) -> int:
        try:
            return self._feature_index_map[int(grain_id)]
        except KeyError as e:
            raise ValueError(f"grain_id={grain_id} not in FeatureIds.") from e

    def search_nn_by_index(self, feature_index: int) -> np.ndarray:
        i = int(feature_index)
        starts = self._neighbor_starts
        if i < 0 or i >= len(starts):
            raise IndexError(f"feature_index out of range: {i}")
        st = int(starts[i])
        en = st + int(self.Num_NN[i])
        return self.Num_list[st:en]

    def search_nn_by_id(self, grain_id: int) -> np.ndarray:
        """Preferred: get neighbors by real grain ID."""
        return self.search_nn_by_index(self._index_from_id(grain_id))

    # Legacy alias: older code treated grain_ID as an index. Here we ensure it's the real ID.
    def search_NN(self, grain_ID: int) -> np.ndarray:
        return self.search_nn_by_id(grain_ID)

    # ---------- Euler / coordinates ----------
    def search_avg_Euler(self, grain_ID: int) -> np.ndarray:
        mask = self.fid == grain_ID
        eulers = self.eul[mask]
        if eulers.size == 0:
            raise ValueError(f"grain_id={grain_ID} not found.")
        return eulers.mean(axis=0)

    def from_ID_to_D(self, Grain_ID: int) -> pd.DataFrame:
        """
        Return a DataFrame of voxel coordinates for the given grain ID.
        The columns are named ['HZ','HY','HX'] to match downstream expectations,
        although the index array is produced in the order [Z, Y, X].
        """
        idx = np.argwhere(self.GrainId == Grain_ID)  # 3 or 4 columns
        df = pd.DataFrame(idx[:, :3], columns=["HZ", "HY", "HX"])
        df["ID"] = int(Grain_ID)
        df = df.sort_values(by=["HZ", "HY", "HX"], ignore_index=True)
        return df.astype({"HZ": "int32", "HY": "int32", "HX": "int32", "ID": "int32"})

    def get_grain_size(self, Grain_ID: int) -> int:
        return int((self.fid == Grain_ID).sum())

    def from_IDs_to_Ds(self, Grain_IDs: List[int]) -> pd.DataFrame:
        parts = [self.from_ID_to_D(g) for g in Grain_IDs]
        return pd.concat(parts, ignore_index=True)

    # ---------- Voxel/volume queries ----------
    def find_volume_grain_ID(
        self,
        HX_range: Tuple[int, int],
        HY_range: Tuple[int, int],
        HZ_range: Tuple[int, int],
        return_count: bool = False,
    ):
        """
        Vectorized crop by H ranges and return unique grain IDs inside.
        If `return_count=True`, also return the counts per grain ID.
        """
        xs = np.arange(HX_range[0], HX_range[1] + 1, dtype=int)
        ys = np.arange(HY_range[0], HY_range[1] + 1, dtype=int)
        zs = np.arange(HZ_range[0], HZ_range[1] + 1, dtype=int)

        if self.GrainId.ndim == 3:
            sub = self.GrainId[np.ix_(zs, ys, xs)]
        elif self.GrainId.ndim == 4:
            sub = self.GrainId[np.ix_(zs, ys, xs, np.array([0]))][..., 0]
        else:
            raise ValueError(f"Unexpected GrainId ndim={self.GrainId.ndim}")

        flat = sub.reshape(-1)
        if not return_count:
            return np.unique(flat)
        else:
            uniq, counts = np.unique(flat, return_counts=True)
            return uniq, counts

    # ---------- Region extension & labeling ----------
    def generate_extend_cube(self, copy_size: int) -> np.ndarray:
        """
        Generate a normalized offset cube of side lengths:
          copy_size=1 -> 3×3×3, copy_size=2 -> 5×5×5, ...
        Returned offsets are divided by `copy_size`.
        """
        scale = (copy_size - 1) / 2
        x, y, z = np.mgrid[-scale : scale + 1, -scale : scale + 1, -scale : scale + 1]
        pos = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
        return pos / copy_size

    def generate_search_cube(self, copy_size: int) -> np.ndarray:
        """
        Generate a (slightly) larger normalized cube for neighborhood search.
        If `copy_size` is even, use scale = copy_size/2 - 1; else use (copy_size-1)/2;
        then expand by ±1 in each axis to include a halo layer.
        Returned offsets are divided by `copy_size`.
        """
        if copy_size % 2 == 0:
            scale = (copy_size / 2) - 1
        else:
            scale = (copy_size - 1) / 2
        x, y, z = np.mgrid[-scale - 1 : scale + 2, -scale - 1 : scale + 2, -scale - 1 : scale + 2]
        pos = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
        return pos / copy_size

    def Extent_Out_data(self, Out_: np.ndarray, unit_extend_ratio: int) -> pd.DataFrame:
        """
        Build an extended set of voxels around the input array.
        The input `Out_` must be an ndarray with columns [HZ, HY, HX, margin-ID, grain-ID].
        Returns a DataFrame with the same columns after applying the extension offsets.
        """
        Unit_cube = self.generate_extend_cube(unit_extend_ratio)
        Extend_Out = np.repeat(Out_, len(Unit_cube), axis=0)
        Extend_Unit = np.tile(Unit_cube, (len(Out_), 1))
        Extend_Out_XYZ = Extend_Out[:, [0, 1, 2]]
        Extend_Out_result = Extend_Out_XYZ + Extend_Unit
        Extend_Out_info = np.hstack((Extend_Out_result, Extend_Out[:, [3, 4]]))
        return pd.DataFrame(Extend_Out_info, columns=["HZ", "HY", "HX", "margin-ID", "grain-ID"])

    def Search_Out_data(self, Out_df: pd.DataFrame, unit_search_ratio: int) -> np.ndarray:
        """
        Given a DataFrame `Out_df` with at least ['HX','HY','HZ'],
        return the search coordinates of shape (len(Out_df)*len(cube))×3 by combining
        each voxel with offsets from `generate_search_cube(unit_search_ratio)`.
        """
        Unit_cube = self.generate_search_cube(unit_search_ratio)
        Out_XYZ = Out_df[["HX", "HY", "HZ"]].to_numpy()
        Extend_Out = np.repeat(Out_XYZ, len(Unit_cube), axis=0)
        Extend_Unit = np.tile(Unit_cube, (len(Out_XYZ), 1))
        return Extend_Out + Extend_Unit

    def find_grain_NN_with_out(self, grain_ID: int, with_diagonal: bool = False) -> pd.DataFrame:
        """
        Compute a margin field around the grain voxels:
        margin-ID: 0 = interior, 1 = outer shell, 2 = inner shell (touching other-grain neighborhood).
        Also assign the appropriate 'grain-ID' to each voxel (0 → this grain; 1/2 → neighbor grain ID).
        """
        # Neighborhood offsets
        if not with_diagonal:
            Unit_cube = np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
            )
            Unit_cube_no_origin = Unit_cube[1:]
        else:
            Unit_cube = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                    [1, 1, 0],
                    [1, -1, 0],
                    [-1, 1, 0],
                    [-1, -1, 0],
                    [1, 0, 1],
                    [1, 0, -1],
                    [-1, 0, 1],
                    [-1, 0, -1],
                    [0, 1, 1],
                    [0, 1, -1],
                    [0, -1, 1],
                    [0, -1, -1],
                    [1, 1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                    [1, 1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [1, -1, -1],
                    [-1, -1, -1],
                ]
            )
            Unit_cube_no_origin = Unit_cube[1:]

        Out = self.from_ID_to_D(grain_ID)
        Out_XYZ = Out[["HZ", "HY", "HX"]].to_numpy()

        # Expand by one layer and mark contour
        Cube_out = _unit_vec(Out_XYZ, Unit_cube)
        Cube_out_df = pd.DataFrame(np.unique(Cube_out, axis=0), columns=["HZ", "HY", "HX"])

        df_all = Cube_out_df.merge(
            Out[["HZ", "HY", "HX"]], on=["HZ", "HY", "HX"], how="left", indicator=True
        )
        # 1 = outer shell, 0 = interior
        df_all["vertice"] = df_all["_merge"].map({"both": 0, "left_only": 1}).astype(int)
        df_all = df_all.drop(columns=["_merge"])

        other_grain_XYZ = df_all[df_all["vertice"] == 1][["HZ", "HY", "HX"]]
        this_grain_XYZ = df_all[df_all["vertice"] == 0][["HZ", "HY", "HX"]]

        other_grain_cube = _unit_vec(other_grain_XYZ.to_numpy(), Unit_cube_no_origin)
        Other_grain_cube = pd.DataFrame(
            np.unique(other_grain_cube, axis=0), columns=["HZ", "HY", "HX"]
        )

        inner_margin = Other_grain_cube.merge(this_grain_XYZ, on=["HZ", "HY", "HX"], how="inner")

        df_with_margin = inner_margin.merge(
            df_all, on=["HZ", "HY", "HX"], how="right", indicator=True
        )
        # 2 = inner shell, 1 = outer shell, 0 = non-boundary
        df_with_margin["margin"] = (
            df_with_margin["_merge"].map({"both": 2, "right_only": 0}).astype(int)
        )
        df_with_margin["margin-ID"] = (df_with_margin["margin"] + df_with_margin["vertice"]).astype(
            int
        )
        df_with_margin = df_with_margin.drop(columns=["_merge", "margin", "vertice"])

        # Clip to volume bounds
        m = (
            (df_with_margin["HX"] >= 0)
            & (df_with_margin["HY"] >= 0)
            & (df_with_margin["HZ"] >= 0)
            & (df_with_margin["HX"] < self.HX_lim)
            & (df_with_margin["HY"] < self.HY_lim)
            & (df_with_margin["HZ"] < self.HZ_lim)
        )
        df_with_margin = df_with_margin[m]

        # Assign grain-ID by margin-ID:
        #   margin==0 → this grain
        #   margin==1/2 → fetch real neighbor id from GrainId volume
        z = df_with_margin["HZ"].to_numpy(dtype=int)
        y = df_with_margin["HY"].to_numpy(dtype=int)
        x = df_with_margin["HX"].to_numpy(dtype=int)
        neighbor_ids = _safe_voxel_id(self.GrainId, z, y, x).astype(int)

        margin = df_with_margin["margin-ID"].to_numpy(dtype=int)
        df_with_margin["grain-ID"] = np.where(margin == 0, int(grain_ID), neighbor_ids)

        return df_with_margin[["HZ", "HY", "HX", "margin-ID", "grain-ID"]]

    def renew_outer_margin_ID(
        self, Extend_Out_: pd.DataFrame, unit_search_ratio: int, extend_ID: bool = False
    ) -> pd.DataFrame:
        """
        Recompute outer margin-ID by performing a local neighborhood vote with a
        search cube of size `unit_search_ratio`.
        If `extend_ID=True`, write the result to column 'extend-ID' instead of 'margin-ID'.
        """
        Extend_Out_df = Extend_Out_.copy()
        Extend_Out_M1 = Extend_Out_df[Extend_Out_df["margin-ID"] == 1].copy()
        Search_XYZ = self.Search_Out_data(Extend_Out_M1, unit_search_ratio)
        Search_XYZ_df = pd.DataFrame(Search_XYZ, columns=["HX", "HY", "HZ"])

        # Normalize, scale, and round to 2 decimals to match grid cells robustly
        for df in (Extend_Out_df, Search_XYZ_df):
            df["HX"] *= unit_search_ratio
            df["HY"] *= unit_search_ratio
            df["HZ"] *= unit_search_ratio
            df[["HX", "HY", "HZ"]] = df[["HX", "HY", "HZ"]].round(2)

        try_merge = Search_XYZ_df.merge(Extend_Out_df, how="left", on=["HX", "HY", "HZ"]).fillna(1)

        number_on_length = unit_search_ratio + (1 if unit_search_ratio % 2 == 0 else 2)
        number_in_cube = number_on_length**3
        group_ids = np.arange(len(try_merge)) // number_in_cube
        margin_id_array = try_merge["margin-ID"].to_numpy(dtype=float)
        mask = (margin_id_array != 1).astype(int)
        extend_ID_count = np.bincount(group_ids, weights=mask)

        Extend_Out_M1_new = Extend_Out_[Extend_Out_["margin-ID"] == 1].copy()
        threshold = (number_in_cube + 1) / 2
        if not extend_ID:
            rewrite_ID_count = np.where(extend_ID_count >= threshold, 2, 1)
            Extend_Out_M1_new["margin-ID"] = rewrite_ID_count
        else:
            extend_ID_count = np.where(extend_ID_count >= threshold, 3, 5)
            Extend_Out_M1_new["extend-ID"] = extend_ID_count

        return Extend_Out_M1_new

    def renew_inner_margin_ID(
        self, Extend_Out_: pd.DataFrame, unit_search_ratio: int, extend_ID: bool = False
    ) -> pd.DataFrame:
        """
        Recompute inner margin-ID using the same neighborhood voting scheme as outer margin.
        If `extend_ID=True`, write the result to 'extend-ID'.
        """
        Extend_Out_df = Extend_Out_.copy()
        Extend_Out_M2 = Extend_Out_df[Extend_Out_df["margin-ID"] == 2].copy()
        Search_XYZ = self.Search_Out_data(Extend_Out_M2, unit_search_ratio)
        Search_XYZ_df = pd.DataFrame(Search_XYZ, columns=["HX", "HY", "HZ"])

        for df in (Extend_Out_df, Search_XYZ_df):
            df["HX"] *= unit_search_ratio
            df["HY"] *= unit_search_ratio
            df["HZ"] *= unit_search_ratio
            df[["HX", "HY", "HZ"]] = df[["HX", "HY", "HZ"]].round(2)

        try_merge = Search_XYZ_df.merge(Extend_Out_df, how="left", on=["HX", "HY", "HZ"]).fillna(1)

        number_on_length = unit_search_ratio + (1 if unit_search_ratio % 2 == 0 else 2)
        number_in_cube = number_on_length**3
        group_ids = np.arange(len(try_merge)) // number_in_cube
        margin_id_array = try_merge["margin-ID"].to_numpy(dtype=float)
        mask = (margin_id_array != 1).astype(int)
        extend_ID_count = np.bincount(group_ids, weights=mask)

        Extend_Out_M2_new = Extend_Out_[Extend_Out_["margin-ID"] == 2].copy()
        threshold = (number_in_cube + 1) / 2
        if not extend_ID:
            rewrite_ID_count = np.where(extend_ID_count >= threshold, 2, 1)
            Extend_Out_M2_new["margin-ID"] = rewrite_ID_count
        else:
            extend_ID_count = np.where(extend_ID_count >= threshold, 3, 5)
            Extend_Out_M2_new["extend-ID"] = extend_ID_count

        return Extend_Out_M2_new

    def get_extend_Out_(self, Out_: pd.DataFrame, unit_extend_ratio: int = 3) -> pd.DataFrame:
        """
        Extend a grain region by `unit_extend_ratio`, recompute both outer(1) and inner(2) margins,
        and return a DataFrame with columns ['HZ','HY','HX','margin-ID','grain-ID'].
        Final H coordinates are scaled by `unit_extend_ratio` and rounded to int.
        """
        Extend_Out_ = self.Extent_Out_data(Out_.to_numpy(), unit_extend_ratio)
        New_M1 = self.renew_outer_margin_ID(Extend_Out_, unit_extend_ratio)
        New_M2 = self.renew_inner_margin_ID(Extend_Out_, unit_extend_ratio)
        Old_M0 = Extend_Out_[Extend_Out_["margin-ID"] == 0]
        New_Extend = pd.concat([New_M1, New_M2, Old_M0], ignore_index=True)
        for ax in ("HX", "HY", "HZ"):
            New_Extend[ax] *= unit_extend_ratio
        New_Extend[["HX", "HY", "HZ"]] = New_Extend[["HX", "HY", "HZ"]].round().astype(int)
        return New_Extend[["HZ", "HY", "HX", "margin-ID", "grain-ID"]]

    def renew_outer_margin(self, New_Extend: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct the outer margin (margin-ID=1) based on the extended set `New_Extend`.
        Returns an integer-typed DataFrame with updated margins.
        """
        Grain_inner = New_Extend[New_Extend["margin-ID"] != 1]
        Grain_XYZ = Grain_inner[["HZ", "HY", "HX"]].to_numpy()
        Unit_cube = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        )
        Cube_out = _unit_vec(Grain_XYZ, Unit_cube)
        Cube_out_df = pd.DataFrame(np.unique(Cube_out, axis=0), columns=["HZ", "HY", "HX"])
        Grain_wOut = Cube_out_df.merge(
            Grain_inner[["HZ", "HY", "HX"]], on=["HZ", "HY", "HX"], how="left", indicator=True
        )
        Grain_wOut["margin-ID"] = Grain_wOut["_merge"].map({"both": 0, "left_only": 1}).astype(int)
        Grain_wOut = Grain_wOut.drop(columns=["_merge"])
        Outer_voxel = Grain_wOut[Grain_wOut["margin-ID"] == 1]
        New_outer_margin = Outer_voxel.merge(
            New_Extend, on=["HZ", "HY", "HX", "margin-ID"], how="left"
        ).dropna()
        out_df = pd.concat([Grain_inner, New_outer_margin], ignore_index=True)
        return out_df.round().astype(int)


@dataclass
class VoxelCSVFrame(Frame):
    """
    A variant of Frame implementation:
    - Voxel grid (GrainId, Dimension) comes from voxel-CSV (space-separated header)
    - Grain orientation comes from Dream3D/HDF5, via internal Frame(search_avg_Euler)

    Usage Example
    -------------
    f = VoxelCSVFrame(
        path="/path/to/t0_Meshed_MoreFeature.dream3d",   # h5 / dream3d
        voxel_csv="/path/to/large_voxel_from_mesh.csv",  # new voxel grid CSV
        h5_grain_dset="GrainID",                         # grain-ID dataset name in h5
    )
    """

    voxel_csv: str | Path = None
    x_col: str = "voxel-X"
    y_col: str = "voxel-Y"
    z_col: str = "voxel-Z"
    grain_col: str = "grain-ID"

    h5_grain_dset: str = "GrainID"

    # Record origin and step size used for H coordinate normalization (optional, for future use)
    H_origin: np.ndarray | None = None  # [x0, y0, z0]
    H_step: np.ndarray | None = None  # [dx, dy, dz]

    _h5_frame: Frame | None = None
    _grain_euler_map: Dict[int, np.ndarray] | None = None

    def __post_init__(self):
        dream_path = Path(self.path)
        csv_path = Path(self.voxel_csv) if self.voxel_csv is not None else None

        if not dream_path.exists():
            raise FileNotFoundError(dream_path)
        if csv_path is None or not csv_path.exists():
            raise FileNotFoundError(f"voxel_csv not found: {csv_path}")

        # ---------- 1. Construct orientation helper Frame from h5 ----------
        base_mapping = self.mapping or {
            "GrainId": self.h5_grain_dset,
            "Euler": "EulerAngles",
            "Num_NN": "NumNeighbors",
            "Num_list": "NeighborList",
            "Dimension": "DIMENSIONS",
        }
        prefer = self.prefer_groups or ["CellData", "CellFeatureData", "_SIMPL_GEOMETRY"]

        h5_frame = Frame(
            dream_path,
            prefer_groups=prefer,
            mapping=base_mapping,
        )
        self._h5_frame = h5_frame

        self.Num_NN = h5_frame.Num_NN
        self.Num_list = h5_frame.Num_list

        grain_euler_map: Dict[int, np.ndarray] = {}
        uniq_ids = np.unique(h5_frame.fid)
        for gid in uniq_ids:
            gid = int(gid)
            if gid <= 0:
                continue
            try:
                grain_euler_map[gid] = h5_frame.search_avg_Euler(gid)
            except ValueError:
                grain_euler_map[gid] = np.array([np.nan, np.nan, np.nan], dtype=float)
        self._grain_euler_map = grain_euler_map

        # ---------- 2. Read voxel-CSV and normalize coordinates to continuous indices ----------
        df = pd.read_csv(
            csv_path,
            sep=r"\s+",  # Key: space / any whitespace separator
            comment="#",
            engine="python",
        )

        required_cols = [self.grain_col, self.x_col, self.y_col, self.z_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"voxel CSV missing columns: {missing}")

        grain_id = df[self.grain_col].astype(int).to_numpy()

        def _normalize_axis(vals: np.ndarray):
            vals = vals.astype(float)
            uniq = np.unique(vals)
            uniq = uniq[~np.isnan(uniq)]
            if uniq.size == 0:
                raise ValueError("Axis has no valid values")
            base = float(uniq.min())
            if uniq.size == 1:
                step = 1.0
            else:
                diffs = np.diff(np.sort(uniq))
                diffs = diffs[diffs > 1e-8]
                step = float(diffs.min()) if diffs.size > 0 else 1.0
            idx = np.rint((vals - base) / step).astype(int)
            return idx, base, step

        gx_raw = df[self.x_col].to_numpy()
        gy_raw = df[self.y_col].to_numpy()
        gz_raw = df[self.z_col].to_numpy()

        hx, x0, dx = _normalize_axis(gx_raw)
        hy, y0, dy = _normalize_axis(gy_raw)
        hz, z0, dz = _normalize_axis(gz_raw)

        self.H_origin = np.array([x0, y0, z0], dtype=float)
        self.H_step = np.array([dx, dy, dz], dtype=float)

        hx_lim = int(hx.max()) + 1
        hy_lim = int(hy.max()) + 1
        hz_lim = int(hz.max()) + 1
        self.Dimension = np.array([hx_lim, hy_lim, hz_lim], dtype=int)

        grid = np.zeros((hz_lim, hy_lim, hx_lim), dtype=int)
        grid[hz, hy, hx] = grain_id
        self.GrainId = grid

        # ---------- 3. Expand Euler grid ----------
        euler_grid = np.zeros((hz_lim, hy_lim, hx_lim, 3), dtype=float)
        for gid, eul in grain_euler_map.items():
            mask = grid == gid
            if not np.any(mask):
                continue
            euler_grid[mask] = eul
        self.Euler = euler_grid

        # ---------- 4. Derived fields: fid / eul / feature_index_map ----------
        self.fid = self.GrainId.flatten()
        self.eul = self.Euler.reshape(-1, 3)

        uniq_ids_grid = np.unique(self.fid)
        self._feature_index_map = {int(fid): i for i, fid in enumerate(uniq_ids_grid)}

    def search_avg_Euler(self, grain_ID: int) -> np.ndarray:
        gid = int(grain_ID)
        if self._grain_euler_map is None:
            raise RuntimeError("grain→Euler map not initialized.")
        try:
            return self._grain_euler_map[gid]
        except KeyError as e:
            raise ValueError(f"grain_id={gid} not found in HDF5 orientation map.") from e
