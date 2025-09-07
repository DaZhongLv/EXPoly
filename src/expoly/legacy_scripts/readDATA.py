import pandas as pd
import h5py as hdf
import numpy as np
import general_func
from typing import List, Optional



read_file_path = "/Users/lvmeizhong/Downloads/D3D_HighlyTextured/Alpoly_textured.dream3d"

def find_dataset_keys(f: hdf.File, target_name: str,
                      prefer_groups: Optional[List[str]] = None) -> List[str]:
    """
    在打开的 HDF5 文件对象 f 中查找名为 target_name 的数据集，返回从根到该数据集的键序列。
    若有多个同名数据集，可用 prefer_groups 指定优先组（按优先顺序匹配子串）。
    例如 prefer_groups = ["CellData", "Grain Data", "FeatureData"]。
    """
    hits = []  # list of key-seq, e.g., ["DataContainers","ImageDataContainer","CellData","FeatureIds"]

    def visit(name, obj):
        nonlocal hits
        if isinstance(obj, hdf.Dataset):
            if name.split('/')[-1].lower() == target_name.lower():
                keys = [k for k in name.split('/') if k]  # drop leading empty
                hits.append(keys)

    f.visititems(visit)
    if not hits:
        raise KeyError(f"Dataset named '{target_name}' not found.")

    # 选择最佳命中
    def score(keys: List[str]) -> tuple:
        # (prefer_rank, path_depth)
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

def make_index_expr(keys: List[str], var: str = "f") -> str:
    """把键序列转成 f['A']['B']['C'] 形式（不含 [()]）"""
    return var + "".join(f"['{k}']" for k in keys)

def gen_assign_lines(f: hdf.File,
                     attr_to_dataset: dict,
                     prefer_groups: Optional[List[str]] = None,
                     self_name: str = "self",
                     file_var: str = "f") -> str:
    """
    根据 {属性名: 数据集basename} 生成多行赋值语句：
        self.Attr = f['...']['...'][()]
    """
    lines = []
    for attr, ds_name in attr_to_dataset.items():
        keys = find_dataset_keys(f, ds_name, prefer_groups=prefer_groups)
        expr = make_index_expr(keys, var=file_var) + "[()]"
        lines.append(f"{self_name}.{attr} = {expr}")
    return "\n".join(lines)


with hdf.File(read_file_path, "r") as f:
    mapping = {
        "GrainId": "FeatureIds",
        "Euler":   "EulerAngles",
        "IPF":     "IPFColor",
        "Num_NN":  "NumNeighbors",
        "Num_list": "NeighborList",
        "Dimension": "DIMENSIONS"
    }
    # prefer = ["CellData", "Grain Data", "CellFeatureData"]  # 可按你习惯调整优先顺序
    code = gen_assign_lines(f, mapping,
                            self_name="self", file_var="f")
    # print(code)

def assign_fields(f: hdf.File, self_obj, mapping: dict, prefer_groups=None):
    """
    在 HDF5 文件 f 中查找指定数据集并赋值到 self_obj 属性。

    Parameters
    ----------
    f : h5py.File
        已经打开的 HDF5 文件
    self_obj : object
        要把属性赋值到的对象（通常是 self）
    mapping : dict
        { "属性名": "数据集basename" } 例如:
        {
          "GrainId": "FeatureIds",
          "Euler":   "EulerAngles",
          "IPF":     "IPFColor",
          "CI":      "Confidence Index",
          "Num_NN":  "NumNeighbors",
          "Num_list":"NeighborList",
        }
    prefer_groups : list[str], optional
        优先选择的 group，例如 ["CellData", "CellFeatureData"]
    """
    for attr, ds_basename in mapping.items():
        # 用前面写过的 find_dataset_keys 去找完整路径
        keys = find_dataset_keys(f, ds_basename, prefer_groups=prefer_groups)
        obj = f
        for k in keys:
            obj = obj[k]
        setattr(self_obj, attr, obj[()])  # 真正把数据放到 self.<attr>


np.set_printoptions(threshold=100)
from scipy import spatial
import time


class Frame(object):
    def __init__(self, current_frame=None):
        self.frame = current_frame

        if current_frame is None or isinstance(current_frame, str) :
            file_path = read_file_path
        else:
            file_path = f"/Users/lvmeizhong/Desktop/EXPoly/Ni_HEDM_Data/An{current_frame}new6.dream3d"

        with hdf.File(file_path, "r") as f:
            mapping = {
                "GrainId": "FeatureIds",
                "Euler": "EulerAngles",
                "IPF": "IPFColor",
                "Num_NN": "NumNeighbors",
                "Num_list": "NeighborList",
                "Dimension": "DIMENSIONS"
            }
            assign_fields(f, self, mapping)

        self.HX_lim = list(self.Dimension)[0]
        self.HY_lim = list(self.Dimension)[1]
        self.HZ_lim = list(self.Dimension)[2]

        self.fid = self.GrainId.flatten()
        self.eul = self.Euler.reshape(-1, 3)

    def search_avg_Euler(self, grain_ID):
        mask = self.fid == grain_ID
        eulers_for_feature = self.eul[mask]
        return eulers_for_feature.mean(axis=0)

    # @nb.njit(fastmath=True)
    def search_NN(self, grain_ID):
        NNN_sum = 0
        from_pos = 0
        end_pos = 0
        for i in range(grain_ID + 1):
            NNN_sum += self.Num_NN[i][0]
            if i == grain_ID - 1:
                from_pos = NNN_sum
            if i == grain_ID:
                end_pos = NNN_sum
        return self.Num_list[from_pos:end_pos]


    def from_ID_to_D(self, Grain_ID):
        Mask_grainID = pd.DataFrame(np.argwhere(self.GrainId == Grain_ID), columns=['HZ', 'HY', 'HX', 'HT'])
        Mask_grainID = Mask_grainID.drop(columns='HT')
        Mask_grainID.sort_values(by=['HZ', 'HX'])
        Mask_grainID['ID'] = Grain_ID
        return Mask_grainID

    def get_grain_size(self, Grain_ID):
        Grain_D = self.from_ID_to_D(Grain_ID)
        return len(Grain_D)

    def from_IDs_to_Ds(self, Grain_IDs):
        # pd.DataFrame(np.argwhere(np.isin(GrainId, [1083, 1056, 1000])), columns=['HX', 'HY', 'HZ', 'HT'])
        save_content = []
        for grain_ID in Grain_IDs:
            this_df = self.from_ID_to_D(grain_ID)
            save_content.append(this_df)
        all_df = pd.concat(save_content)
        return all_df

    def get_neighborhood(self, grain_ID, neighborhood):
        all_grain = {}
        its_NN = self.search_NN(grain_ID)
        all_grain[0] = [grain_ID]
        all_grain[1] = its_NN

        now_neighborhood = 1
        while now_neighborhood <= neighborhood:
            now_check_nn_list = all_grain[now_neighborhood]
            this_new_neighborhood = []
            for each_NN in now_check_nn_list:
                this_grain_NN = self.search_NN(each_NN)
                this_new_neighborhood.extend(this_grain_NN)

            this_new_neighborhood_unique = np.unique(this_new_neighborhood)
            real_this_new_neighborhood = np.setdiff1d(this_new_neighborhood_unique,
                                                      [x for v in all_grain.values() for x in v])
            now_neighborhood += 1
            all_grain[now_neighborhood] = real_this_new_neighborhood
        return all_grain

    def find_volume_grain_ID(self, HX_range, HY_range, HZ_range, return_count=False):
        st = time.process_time()
        X_dim = np.arange(HX_range[0], HX_range[1] + 1, 1, dtype=int)
        Y_dim = np.arange(HY_range[0], HY_range[1] + 1, 1, dtype=int)
        Z_dim = np.arange(HZ_range[0], HZ_range[1] + 1, 1, dtype=int)
        cube_struct = [[i, j, k] for i in X_dim
                       for j in Y_dim
                       for k in Z_dim]
        Grain_ID_record = []
        for each_pos in cube_struct:
            HX = each_pos[0]
            HY = each_pos[1]
            HZ = each_pos[2]
            id = self.GrainId[HZ][HY][HX]
            Grain_ID_record.extend(id)
        if return_count == False:
            consider_grain = np.unique(Grain_ID_record)
            et = time.process_time()
            dur = et - st
            print('CPU Execution time in find volume grain ID:', dur, 'seconds')
            return consider_grain
        else:
            consider_grain, counts = np.unique(Grain_ID_record, return_counts=True)
            et = time.process_time()
            dur = et - st
            print('CPU Execution time in find volume grain ID:', dur, 'seconds')
            return consider_grain, counts

    def find_grain_NN_with_out(self, grain_ID, with_diagonal=False):
        def find_grain_ID_vec(array):
            if array[3] == 0:
                return grain_ID
            else:
                return self.GrainId[array[0]][array[1]][array[2]][0]           ###### gai shun xu?

        if with_diagonal == False:  # no diagonal for true NN in hdf5
            Unit_cube = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
            Unit_cube_no_origin = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            Unit_cube = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],
                                  [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                                  [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                                  [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
                                  [1, 1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, -1],
                                  [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]
                                  ])
            Unit_cube_no_origin = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],
                                            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                                            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                                            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
                                            [1, 1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, -1],
                                            [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1],
                                            ])
        Out = self.from_ID_to_D(grain_ID)  # largest = 357, smallest = 442
        Out_XYZ = Out[['HZ', 'HY', 'HX']]

        st = time.process_time()

        Cube_out = general_func.Unit_vec(Out_XYZ.to_numpy(), Unit_cube)
        Cube_out_df = pd.DataFrame(np.unique(Cube_out.reshape(-1, Cube_out.shape[-1]), axis=0),
                                   columns=['HZ', 'HY', 'HX'])

        df_all = Cube_out_df.merge(Out_XYZ, on=['HZ', 'HY', 'HX'],
                                   how='left', indicator=True)

        df_all['vertice'] = df_all['_merge'].map({'both': 0, 'left_only': 1})  # 1 outer margin, 0 non margin
        df_all = df_all.drop(columns=['_merge'])

        other_grain_XYZ = df_all[df_all['vertice'] == 1][['HZ', 'HY', 'HX']]
        this_grain_XYZ = df_all[df_all['vertice'] == 0][['HZ', 'HY', 'HX']]

        other_grain_cube = general_func.Unit_vec(other_grain_XYZ.to_numpy(), Unit_cube_no_origin)
        Other_grain_cube = pd.DataFrame(np.unique(other_grain_cube.reshape(-1, other_grain_cube.shape[-1]), axis=0),
                                        columns=['HZ', 'HY', 'HX'])

        inner_margin = Other_grain_cube.merge(this_grain_XYZ, on=['HZ', 'HY', 'HX'], how='inner')

        df_with_margin = inner_margin.merge(df_all, on=['HZ', 'HY', 'HX'],
                                            how='right', indicator=True)

        df_with_margin['margin'] = df_with_margin['_merge'].map(
            {'both': 2, 'right_only': 0})  # 2 outer margin, 0 non margin
        df_with_margin['margin-ID'] = df_with_margin['margin'] + df_with_margin['vertice']

        df_with_margin = df_with_margin.drop(columns=['_merge', 'margin', 'vertice'])
        df_with_margin = df_with_margin[
            (df_with_margin['HX'] < self.HX_lim) & (df_with_margin['HY'] < self.HY_lim) & (df_with_margin['HZ'] < self.HZ_lim)]
        vfunc = np.vectorize(find_grain_ID_vec, signature='(n)->()')
        try_this_out = vfunc(df_with_margin.astype(int).to_numpy())
        et = time.process_time()
        dur = et - st
        print('CPU Execution time in find outer NN:', dur, 'seconds')

        df_with_margin['grain-ID'] = try_this_out
        return df_with_margin

    def generate_extend_cube(self, copy_size):
        scale_size = (copy_size - 1) / 2
        # copy size = 1, 3x3 Rubik's Cube; copy size = 2, 5x3 Rubik's Cube
        # Unit_cube_scale = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],
        # #                                   [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        # #                                   [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        # #                                   [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        # #                                   [1, 1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, -1],
        # #                                   [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]
        # #                                   ])
        x, y, z = np.mgrid[-scale_size:scale_size + 1, -scale_size:scale_size + 1, -scale_size:scale_size + 1]
        positions = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
        return positions / copy_size

    def generate_search_cube(self, copy_size):
        # copy size = 1, 3x3 Rubik's Cube; copy size = 2, 5x3 Rubik's Cube

        if copy_size % 2 == 0:
            scale_size = (copy_size / 2) - 1
        else:
            scale_size = (copy_size - 1) / 2
        x, y, z = np.mgrid[-scale_size - 1:scale_size + 1 + 1, -scale_size - 1:scale_size + 1 + 1,
                  -scale_size - 1:scale_size + 1 + 1]
        positions = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
        return positions / copy_size

    def Extent_Out_data(self, Out_, unit_extend_ratio):
        ## default the Out_ column name: HZ,HY,HX,margin-ID,grain-ID, used the numpy icol below
        ## and also scale up to make each voxel length as 1
        Unit_cube = self.generate_extend_cube(unit_extend_ratio)
        # print(Unit_cube)
        Extend_Out = np.repeat(Out_, len(Unit_cube), axis=0)
        Extend_Unit = np.tile(Unit_cube, (len(Out_), 1)) * 1
        Extend_Out_XYZ = Extend_Out[:, [0, 1, 2]]
        Extend_Out_result = np.add(Extend_Out_XYZ, Extend_Unit)

        Extend_Out_info = np.hstack((Extend_Out_result, Extend_Out[:, [3, 4]]))
        Extend_Out_df = pd.DataFrame(Extend_Out_info, columns=['HZ', 'HY', 'HX', 'margin-ID', 'grain-ID'])
        return Extend_Out_df

    def Search_Out_data(self, Out_, unit_search_ratio):
        ## default the Out_ column name: HZ,HY,HX,margin-ID,grain-ID, used the numpy icol below
        ## and also scale up to make each voxel length as 1
        Unit_cube = self.generate_search_cube(unit_search_ratio)
        # print(Unit_cube)
        Out_XYZ = Out_[['HX', 'HY', 'HZ']]
        Extend_Out = np.repeat(Out_XYZ, len(Unit_cube), axis=0)
        Extend_Unit = np.tile(Unit_cube, (len(Out_), 1)) * 1
        # Extend_Out_XYZ = Extend_Out[:, [0,1,2]]
        Extend_Out_result = np.add(Extend_Out, Extend_Unit)
        return Extend_Out_result


    def renew_outer_margin_ID(self, Extend_Out_, unit_search_ratio, extend_ID=False):
        Extend_Out_df = Extend_Out_.copy()
        Extend_Out_M1 = Extend_Out_df[Extend_Out_df['margin-ID'] == 1].copy()
        Search_XYZ = self.Search_Out_data(Extend_Out_M1, unit_search_ratio)
        Search_XYZ_df = pd.DataFrame(Search_XYZ, columns=['HX', 'HY', 'HZ'])

        Extend_Out_df['HX'] *= unit_search_ratio  # unit_search_ratio
        Extend_Out_df['HY'] *= unit_search_ratio
        Extend_Out_df['HZ'] *= unit_search_ratio
        Extend_Out_df = Extend_Out_df.round(2)
        Search_XYZ_df['HX'] *= unit_search_ratio  # unit_search_ratio
        Search_XYZ_df['HY'] *= unit_search_ratio
        Search_XYZ_df['HZ'] *= unit_search_ratio
        Search_XYZ_df = Search_XYZ_df.round(2)
        try_merge = Search_XYZ_df.merge(Extend_Out_df, how='left', on=['HX', 'HY', 'HZ']).fillna(
            1)  # no matter the grain-ID, nan should only be other NN grain voxel
        number_on_length = unit_search_ratio + (1 if unit_search_ratio % 2 == 0 else 2)
        number_in_cube = number_on_length ** 3

        group_ids = np.arange(len(try_merge)) // number_in_cube

        margin_id_array = try_merge['margin-ID'].values
        mask = (margin_id_array != 1).astype(int)
        extend_ID_count = np.bincount(group_ids, weights=mask)

        Extend_Out_M1_new = Extend_Out_[Extend_Out_['margin-ID'] == 1].copy()
        threshold = (number_in_cube + 1) / 2
        if not extend_ID:
            rewrite_ID_count = np.where(extend_ID_count >= threshold, 2, 1)
            Extend_Out_M1_new['margin-ID'] = rewrite_ID_count
        else:
            extend_ID_count = np.where(extend_ID_count >= threshold, 3, 5)
            Extend_Out_M1_new['extend-ID'] = extend_ID_count
            # 6 means outer margin -> outer margin, 4 means outer magin -> inner margin

        del try_merge
        del Search_XYZ
        del Extend_Out_df
        del Extend_Out_M1

        return Extend_Out_M1_new


    def renew_inner_margin_ID(self, Extend_Out_, unit_search_ratio, extend_ID=False):
        Extend_Out_df = Extend_Out_.copy()
        Extend_Out_M2 = Extend_Out_df[Extend_Out_df['margin-ID'] == 2]
        Search_XYZ = self.Search_Out_data(Extend_Out_M2, unit_search_ratio)
        Search_XYZ_df = pd.DataFrame(Search_XYZ, columns=['HX', 'HY', 'HZ'])

        Extend_Out_df['HX'] *= unit_search_ratio  # unit_search_ratio
        Extend_Out_df['HY'] *= unit_search_ratio
        Extend_Out_df['HZ'] *= unit_search_ratio
        Extend_Out_df = Extend_Out_df.round(2)
        Search_XYZ_df['HX'] *= unit_search_ratio  # unit_search_ratio
        Search_XYZ_df['HY'] *= unit_search_ratio
        Search_XYZ_df['HZ'] *= unit_search_ratio
        Search_XYZ_df = Search_XYZ_df.round(2)
        try_merge = Search_XYZ_df.merge(Extend_Out_df, how='left', on=['HX', 'HY', 'HZ']).fillna(
            1)  # no matter the grain-ID, nan should only be other NN grain voxel

        number_on_length = unit_search_ratio + (1 if unit_search_ratio % 2 == 0 else 2)
        number_in_cube = number_on_length ** 3

        group_ids = np.arange(len(try_merge)) // number_in_cube
        margin_id_array = try_merge['margin-ID'].values

        mask = (margin_id_array != 1).astype(int)
        extend_ID_count = np.bincount(group_ids, weights=mask)

        Extend_Out_M2_new = Extend_Out_[Extend_Out_['margin-ID'] == 2].copy()
        threshold = (number_in_cube + 1) / 2
        if not extend_ID:
            rewrite_ID_count = np.where(extend_ID_count >= threshold, 2, 1)
            Extend_Out_M2_new['margin-ID'] = rewrite_ID_count
        else:
            extend_ID_count = np.where(extend_ID_count >= threshold, 3, 5)
            Extend_Out_M2_new['extend-ID'] = extend_ID_count
            # 5 means inner margin -> outer margin, 3 means inner magin -> inner margin

        del try_merge
        del Search_XYZ
        del Extend_Out_df
        del Extend_Out_M2

        return Extend_Out_M2_new

    def get_extend_Out_(self, Out_, unit_extend_ratio=3):
        Extend_Out_ = self.Extent_Out_data(Out_, unit_extend_ratio)
        New_M1 = self.renew_outer_margin_ID(Extend_Out_, unit_extend_ratio)
        New_M2 = self.renew_inner_margin_ID(Extend_Out_, unit_extend_ratio)
        Old_M0 = Extend_Out_[Extend_Out_['margin-ID'] == 0]
        New_Extend = pd.concat([New_M1, New_M2, Old_M0])
        New_Extend['HX'] *= unit_extend_ratio
        New_Extend['HY'] *= unit_extend_ratio
        New_Extend['HZ'] *= unit_extend_ratio
        New_Extend = New_Extend.round().astype(int)
        return New_Extend

    def renew_outer_margin(self, New_Extend):
        Grain_inner = New_Extend[New_Extend['margin-ID'] != 1]
        Grain_XYZ = Grain_inner[['HZ', 'HY', 'HX']]
        Unit_cube = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        Cube_out = general_func.Unit_vec(Grain_XYZ.to_numpy(), Unit_cube)
        Cube_out_df = pd.DataFrame(np.unique(Cube_out.reshape(-1, Cube_out.shape[-1]), axis=0),columns=['HZ', 'HY', 'HX'])
        Grain_wOut = Cube_out_df.merge(Grain_XYZ, on=['HZ', 'HY', 'HX'], how='left', indicator=True)
        Grain_wOut['margin-ID'] = Grain_wOut['_merge'].map({'both': 0, 'left_only': 1})  # 1 outer margin, 0 non margin
        Grain_wOut = Grain_wOut.drop(columns=['_merge'])
        Outer_voxel = Grain_wOut[Grain_wOut['margin-ID'] == 1]
        New_outer_margin = Outer_voxel.merge(New_Extend, on=['HZ', 'HY', 'HX', 'margin-ID'], how='left').dropna()
        return pd.concat([Grain_inner, New_outer_margin]).round().astype(int)


class Frame_detail(Frame):
    def __init__(self, current_frame):
        super().__init__(current_frame)

    def find_topo(self, grain_ID):
        Out = self.find_grain_NN_with_out(grain_ID, with_diagonal=True)  # better when finding quadruple,
        Out_XYZ = Out[['HZ', 'HY', 'HX']].to_numpy()
        Out_array = Out.to_numpy()
        inner_margin = Out[Out['margin-ID'] == 2]
        inner_margin_XYZ = inner_margin[['HZ', 'HY', 'HX']].to_numpy()
        inner_atom = Out[Out['margin-ID'] == 0]

        tree = spatial.cKDTree(Out_XYZ)
        index = tree.query_ball_point(inner_margin_XYZ, r=1.45)

        inner_margin_topo = []
        for each_pos in index:
            pos_topo = len(np.unique(Out_array[each_pos, -1]))
            inner_margin_topo.append(pos_topo)

        inner_margin['topo-ID'] = inner_margin_topo
        inner_atom['topo-ID'] = 1

        grain_with_topo = pd.concat([inner_margin, inner_atom])
        return grain_with_topo

    def find_grains_all_edges(self, grain_ID_list):
        each_NN_len = []
        each_NN = []
        for each_grain in grain_ID_list:
            its_NN = self.search_NN(each_grain)
            its_NN_len = len(its_NN)
            each_NN_len.append(its_NN_len)
            each_NN.extend(its_NN)

        this_grain_full = np.repeat(grain_ID_list, each_NN_len)  # this_grain
        other_grain_full = np.array(each_NN)

        all_edges = np.vstack((this_grain_full, other_grain_full)).T
        all_edges.sort(axis=1)
        unique_all_edges = np.unique(all_edges, axis=0)
        return unique_all_edges

    def extract_data(self, array):
        HX = array[0]
        HY = array[1]
        HZ = array[2]
        euler = self.Euler[HZ][HY][HX]
        id = self.GrainId[HZ][HY][HX]
        ipf= self.IPF[HZ][HY][HX]
        # ci = CI[HX][HY][HZ]
        return np.concatenate((array, euler, id, ipf))#xpos,ypos,euler, ci,


    def edge_in_grain_list(self, grain_list):
        # @nb.njit(fastmath=True)
        def isin(val, arr):
            for i in range(len(arr)):
                if arr[i] == val:
                    return True
            return False

        st = time.process_time()
        this_grain_l = []
        other_grain_l = []
        for each_grain in grain_list:
            this_grain_nn = self.search_NN(each_grain)
            for each_grain_nn in this_grain_nn:
                if isin(each_grain_nn, grain_list):
                    this_grain_l.append(each_grain)
                    other_grain_l.append(each_grain_nn)

        this_grain_a = np.array(this_grain_l)
        other_grain_a = np.array(other_grain_l)
        grain_edge_a = np.vstack((this_grain_a, other_grain_a)).T
        grain_edge_a.sort(axis=1)
        unique_all_edges = np.unique(grain_edge_a, axis=0)
        et = time.process_time()
        dur = et - st
        print('CPU Execution time in find edge in grain list:', dur, 'seconds')
        return unique_all_edges

    def get_neighborhood_with_edge(self, grain_ID, neighborhood):
        st = time.process_time()
        all_grain_d = {}
        its_NN = self.search_NN(grain_ID)
        all_grain_d[0] = [grain_ID]
        all_grain_d[1] = its_NN.tolist()

        all_grain = [grain_ID]
        grains_idx = []

        all_grain.extend(its_NN)
        grains_idx.append(len(its_NN))
        now_neighborhood = 1
        while now_neighborhood < neighborhood:
            now_check_nn_list = all_grain_d[now_neighborhood]
            this_new_neighborhood = []
            for each_NN in now_check_nn_list:
                this_grain_NN = self.search_NN(each_NN)
                this_new_neighborhood.extend(this_grain_NN)
                grains_idx.append(len(this_grain_NN))

            all_grain.extend(this_new_neighborhood)
            this_new_neighborhood_unique = np.unique(this_new_neighborhood)
            real_this_new_neighborhood = np.setdiff1d(this_new_neighborhood_unique,
                                                      [x for v in all_grain_d.values() for x in v])
            now_neighborhood += 1
            all_grain_d[now_neighborhood] = real_this_new_neighborhood.tolist()

        extend_nn_list = all_grain[0:len(grains_idx)]
        extend_to_full = np.repeat(extend_nn_list, grains_idx)  # this_grain

        edge_pairs = np.vstack((extend_to_full, all_grain[1:])).T
        edge_pairs.sort(axis=1)
        unique_edge_pairs = np.unique(edge_pairs, axis=0)
        external_need_add = all_grain_d[neighborhood]
        grain_list_refer = general_func.list_flatten([all_grain_d[key] for key in all_grain_d.keys()])

        external_edges_this = []
        external_edges_other = []
        for each_external_grain in external_need_add:
            this_grain_NN = self.search_NN(each_external_grain)
            find_edges = [value for value in this_grain_NN if value in grain_list_refer]
            if len(find_edges) > 0:
                external_edges_other.extend(find_edges)
                external_this_contents = [each_external_grain] * len(find_edges)
                external_edges_this.extend(external_this_contents)

        external_edge_ = np.vstack((external_edges_this, external_edges_other)).T
        external_edge_.sort(axis=1)
        unique_edge_pairs = np.vstack((unique_edge_pairs, external_edge_))
        et = time.process_time()
        dur = et - st
        print('CPU Execution time in create grain NN dict and edges:', dur, 'seconds')
        return all_grain_d, np.unique(unique_edge_pairs, axis=0)

    def pair_misorientation(self, unique_pair, with_hkl = False):
        if with_hkl == False:
            mori = []
            for each_row in unique_pair:
                this_eul = self.search_avg_Euler(int(each_row[0]))
                other_eul = self.search_avg_Euler(int(each_row[1]))
                this_mori = general_func.calculate_misorientation(this_eul, other_eul, general_func.symmetry_operators)

                mori.append(this_mori)

            Pair = pd.DataFrame(unique_pair, columns=['T', 'O'])
            Pair['mori'] = mori
            return Pair
        else:
            mori, h_l,k_l,l_l = [],[],[],[]
            for each_row in unique_pair:
                this_eul = self.search_avg_Euler(int(each_row[0]))
                other_eul = self.search_avg_Euler(int(each_row[1]))
                this_mori, h,k,l = general_func.calculate_misorientation_and_hkl(this_eul, other_eul, general_func.symmetry_operators)
                mori.append(this_mori)
                h_l.append(h)
                k_l.append(k)
                l_l.append(l)

            Pair = pd.DataFrame(unique_pair, columns=['T', 'O'])
            Pair['mori'] = mori
            Pair['h'] = h_l
            Pair['k'] = k_l
            Pair['l'] = l_l
            return Pair


    def selection_sigma(self, unique_pair, sigma_angle, range, size_mini=1000):
        Pair_mori_pre = self.pair_misorientation(unique_pair, with_hkl = True)
        select_mori = Pair_mori_pre[(Pair_mori_pre['mori'] > sigma_angle - range) & (Pair_mori_pre['mori'] < sigma_angle + range)]
        # print(select_mori)
        unique_grain = np.unique(select_mori[['T', 'O']].to_numpy())
        if len(unique_grain) == 0:
            return None
        else:
            vfunc = np.vectorize(self.get_grain_size)
            grain_size = vfunc(unique_grain)
            grain_ID_size = pd.DataFrame(np.vstack((unique_grain, grain_size)).T, columns=['T', 'T_size'])
            Pair_T = select_mori.merge(grain_ID_size, how='left', on='T')
            Pair_T_select = Pair_T[(Pair_T['T_size'] > size_mini)]
            grain_ID_size = grain_ID_size.rename(columns={"T": "O", "T_size": "O_size"})
            Pair_O = Pair_T_select.merge(grain_ID_size, how='left', on='O')
            Pair_O_select = Pair_O[(Pair_O['O_size'] > size_mini)]
            return Pair_O_select

        # grain_size_df = pd.DataFrame(grain_size, columns=['T_size', 'O_size'])
        # Pair = pd.concat([select_mori, grain_size_df], axis=1)
        # print(Pair)



# Frame_0 = Frame_detail(current_frame=0)
if __name__ == '__main__':


    Structure = Frame()
    # print(Structure.FeatureIds)
    # print(Structure.HX_lim)
    # print(Structure.from_ID_to_D(2813))
    # print(Structure.from_ID_to_D(3142))
    # print(Structure.from_ID_to_D(3111))

    print(Structure.search_avg_Euler(1200))


#
# Frame_1 = Frame(current_frame=1)
# Frame_2 = Frame(current_frame=2)
#     Frame_0 = Frame_detail(current_frame=0)

    # unique_ID, counts = Frame_0.find_volume_grain_ID([60, 320], [60, 320], [0, 60], return_count=True)
    # collect_info = np.vstack((unique_ID, counts)).T
    # collect_info_df = pd.DataFrame(collect_info, columns=['unique_ID', 'counts'])
    # collect_info_df.to_csv('/Users/lvmeizhong/Downloads/collect_info.csv')



    # print(Frame_1.find_grain_NN_with_out(123))
    # print(Frame_1.from_ID_to_D(123))



    # print(general_func.get_step_grain_ID(2000, 0, 1, return_all=True))
    # print(general_func.get_step_grain_ID(2551, 1, 0, return_all=True))
    # HZ_range = [0, 60]
    # HX_range = [60, 320]
    # HY_range = [60, 320]

# edge_in_grain_list(self, grain_list)

#     pairs = Frame_1.edge_in_grain_list([25, 122, 195, 268, 316, 422, 426, 542, 636, 743, 841, 869, 1018, 1068, 1118, 1127, 1132, 1173, 1181, 1249, 1320, 1361, 1394, 1555, 1606, 1619, 1635, 1637, 1656, 1683, 1917, 1925, 1952, 2007, 2018, 2053, 2064, 2124, 2199, 2256, 2324, 2328, 2342, 2381, 2429, 2481, 2557, 2590, 2674, 2696, 2741, 2808, 2879, 2927]
# )
    # pairs = Frame_1.edge_in_grain_list([2741, 2443, 2328, 1249, 1683, 268, 2124, 743, 1127, 0])
    # print(pairs)
    #
    # MORI = Frame_1.pair_misorientation(pairs, with_hkl = False)
    # MORI_60 = MORI[MORI['mori'] > 59]
    # print(MORI_60)




    # Grain_ = Frame_0.find_grain_NN_with_out(111)
    # Grain_NN = Frame_0.search_NN(2332)
    # for each_NN in Grain_NN:
    #     this_later_ID = general_func.get_step_grain_ID(each_NN, 0, 1)
    #     print(each_NN, this_later_ID)



#     # pd.options.display.max_rows = 1000
#     # print(Frame_0.search_centroid(510))
#     # print(Frame_0.search_centroid(1506))
#
# # #     S = Frame_0.find_grain_NN_with_out(3, with_diagonal=True)
#     import plotly.express as px
# #     import plotly.graph_objects as go
# #
#     fig = px.scatter_3d(Grain_2332, x='HX', y='HY', z='HZ', color='grain-ID')
#
#     fig.update_layout(scene_aspectmode='data')
#     fig.show()

#########################################################
    # HX_range = [0, 60]
    # HY_range = [180, 270]
    # HZ_range = [180, 270]
    # grain_list = Frame_0.find_volume_grain_ID(HX_range, HY_range, HZ_range)
    # edges = Frame_0.edge_in_grain_list(grain_list)
    # # Frame_0.selection_sigma(edges, 38.94, 0.02, size_mini=1000)
    # # #
    # pd.set_option('display.max_rows', None)
    # select_mori = Frame_0.selection_sigma(edges, 38.94, 0.02, size_mini=500)
    # 5: 36.86[100]; 7: 38.21[111]; 9: 38.94[110]
    # all_mori = Frame_0.pair_misorientation([[680,  2284]], with_hkl=True)
    # print(all_mori)
    # # center series
    # inner = [14, 28, 43, 136, 263, 288, 313, 371, 391, 395, 412, 454, 461, 602, 672, 688, 692, 719, 736, 753, 923, 1007, 1077, 1107, 1147, 1201, 1215, 1355, 1364, 1387, 1394, 1403, 1529, 1567, 1590, 1619, 1664, 1686, 1737, 1788, 1816, 1856, 1932, 1952, 1992, 2038, 2053, 2328, 2400, 2403, 2418, 2533, 2540, 2615, 2684, 2696, 2792, 2798, 2837, 2855, 2884, 2892, 2912, 2972]




    # # #
    # # inner = [47, 69, 72, 104, 113, 122, 164, 166, 183, 237, 314, 529, 531, 713, 801, 811, 840, 915, 945, 986, 993, 995,
    # #          1069, 1084, 1115, 1119, 1127, 1191, 1212, 1262, 1275, 1307, 1386, 1391, 1395, 1424, 1473, 1532, 1562, 1606,
    # #          1609, 1631, 1654, 1672, 1742, 1754, 1838, 1878, 1988, 2025, 2062, 2188, 2190, 2218, 2299, 2332, 2337, 2360,
    # #          2463, 2604, 2627, 2662, 2800, 2805, 2879, 2932, 2940]
    # select_mori = all_mori[(all_mori['mori'] >38.93) & (all_mori['mori'] < 38.95) ]
    # select_mori = select_mori[(select_mori['T'].isin(inner)) & (select_mori['O'].isin(inner))]
    # #
    # print(select_mori)
#########################################################
    # print(len(Frame_0.from_ID_to_D(90)))

    # print(Frame_0.pair_misorientation([[2199, 2357]], with_hkl=True))






