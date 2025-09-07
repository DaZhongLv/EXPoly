import pandas as pd
import numpy as np
np.set_printoptions(threshold=100)
import math
pd.options.mode.chained_assignment = None  # default='warn'

import global_variable
import general_func
import plotly.express as px
from scipy import spatial
from scipy.spatial import ConvexHull, Delaunay
import time
from scipy.spatial.distance import cdist
import plotly.graph_objects as go

def Unit_vec_ratio(Out_XYZ, Unit_cube, ratio):   # Out_XYZ is numpy array
    Extend_Out = np.repeat(Out_XYZ, len(Unit_cube), axis=0)
    Extend_Unit = np.tile(Unit_cube, (len(Out_XYZ), 1)) * ratio
    return np.add(Extend_Out, Extend_Unit)

def Extent_Out_data(Out_, Unit_cube=global_variable.Unit_cube):
    ## default the Out_ column name: HZ,HY,HX,margin-ID,grain-ID, used the numpy icol below
    ## and also scale up to make each voxel length as 1
    Extend_Out = np.repeat(Out_, len(Unit_cube), axis=0)
    Extend_Unit = np.tile(Unit_cube, (len(Out_), 1)) * 1
    Extend_Out_XYZ = Extend_Out[:, [0,1,2]]
    Extend_Out_result = np.add(Extend_Out_XYZ, Extend_Unit) * 3

    Extend_Out_info = np.hstack((Extend_Out_result, Extend_Out[:,[3,4]]))
    Extend_Out_df = pd.DataFrame(Extend_Out_info, columns=['HZ','HY','HX','margin-ID','grain-ID'])
    return  Extend_Out_df

def SC2FCC(data, ratio):
    Unit_FCC = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    return Unit_vec_ratio(data, Unit_FCC, ratio)

def SC2BCC(data, ratio):
    Unit_BCC = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    return Unit_vec_ratio(data, Unit_BCC, ratio)

def SC2DIA(data, ratio):
    Unit_DIA = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]])
    return Unit_vec_ratio(data, Unit_DIA, ratio)

def check_SC2FCC(T_grid):
    T_grid = T_grid.reshape(-1, T_grid.shape[-1])
    atom_ID = np.arange(1, 1 + (len(T_grid)/4), 1, dtype=int)

    atom_ID_repeat = np.repeat(a=atom_ID, repeats=4)
    atom_refer = np.tile(np.array([0, 1, 1, 1]), int((len(T_grid)/4)))

    for_draw = pd.DataFrame(T_grid, columns=['X', 'Y', 'Z'])
    for_draw['atom-ID'] = atom_ID_repeat
    for_draw['atom-refer'] = atom_refer
    return for_draw


def move_to_Hcenter(data, Hcenter):
    data = data.reshape(-1, data.shape[-1])
    data[:, 0] += Hcenter[0]
    data[:, 1] += Hcenter[1]
    data[:, 2] += Hcenter[2]
    return data

def prepare_carve_FCC(Out, ratio, random=True, use_BCC=False, use_DIA=False):  #normal ratio = 1.5, random select ball's center
    grain_ID = Out['ID'][0]
    avg_Eul = global_variable.Frame_.search_avg_Euler(grain_ID)
    R = general_func.eul2rot(avg_Eul)


    HX_min, HX_max = Out['HX'].min(),Out['HX'].max()
    HY_min, HY_max = Out['HY'].min(),Out['HY'].max()
    HZ_min, HZ_max = Out['HZ'].min(),Out['HZ'].max()
    Hcenter = [0.5*(HX_max + HX_min),
               0.5*(HY_max + HY_min),
               0.5*(HZ_max + HZ_min)]
    # Hcenter = np.array(center)
    Crave_R = (math.dist([HX_min, HY_min, HZ_min], [HX_max, HY_max, HZ_max]) / 2) * 1.5   # 1.5 hyperparameter
    one_dim = np.arange(-Crave_R, Crave_R+1, ratio, dtype=float)
    cube_strcut = [[i, j, k] for i in one_dim
                             for j in one_dim
                             for k in one_dim]

    st = time.process_time()
    tree = spatial.cKDTree(cube_strcut)
    if random == True:
        ball_index = tree.query_ball_point(np.random.sample(3)*2, Crave_R, workers=-1)
    else:
        ball_index = tree.query_ball_point([0,0,0], Crave_R, workers=-1)
    ball_struct = np.take(cube_strcut, ball_index, axis=0)



    # one_dim_axis = np.linspace(0, 30, 100)
    # three_dim_zeros = np.zeros((len(one_dim_axis), 3))
    # X_axis = np.copy(three_dim_zeros)
    # X_axis[:, 0] = one_dim_axis
    # Y_axis = np.copy(three_dim_zeros)
    # Y_axis[:, 1] = one_dim_axis
    # Z_axis = np.copy(three_dim_zeros)
    # Z_axis[:, 2] = one_dim_axis

    ######### plot ball struct plot
    # ball_struct_df = pd.DataFrame(ball_struct, columns=['HX', 'HY', 'HZ'])
    # ball_struct_df['HH'] = ball_struct_df['HZ'].apply(lambda x: 1 if x == one_dim[15] else 0)
    # ball_struct_df['HV'] = ball_struct_df['HY'].apply(lambda x: 2 if x == one_dim[15] else 0)
    # ball_struct_df['HP'] = ball_struct_df['HH'] + ball_struct_df['HV']
    # ball_struct_df['HP'] = ball_struct_df['HP'].astype(dtype='str')
    # # real_index = ball_struct_df.index.to_numpy()
    # heritage_HP = ball_struct_df['HP'].to_numpy()
    #
    # color_map = {'0':'rgba(64, 64, 64, 0.4)', '1':'rgba(255, 255, 0, 1)', '2':'rgba(204, 0, 0, 1)', '3':'rgba(0, 204, 0, 1)'}
    # fig = px.scatter_3d(x=ball_struct_df['HX'], y=ball_struct_df['HY'], z=ball_struct_df['HZ'], color=ball_struct_df['HP'],
    #                     opacity=0.2, color_discrete_map=color_map)
    # fig.add_trace(go.Scatter3d(x=X_axis[:, 0], y=X_axis[:, 1], z=X_axis[:, 2], mode='lines',
    #                            line=dict(color='rgba(0, 153, 0, 1)', width=10)))
    # fig.add_trace(go.Scatter3d(x=Y_axis[:, 0], y=Y_axis[:, 1], z=Y_axis[:, 2], mode='lines',
    #                            line=dict(color='rgba(255, 255, 0, 1)', width=10)))
    # fig.add_trace(go.Scatter3d(x=Z_axis[:, 0], y=Z_axis[:, 1], z=Z_axis[:, 2], mode='lines',
    #                            line=dict(color='rgba(204, 0, 0, 1)', width=10)))
    # #
    # camera_params = dict(
    #     up=dict(x=0, y=0, z=1),
    #     center=dict(x=0, y=0, z=0),
    #     eye=dict(x=1.25, y=0, z=0)
    # )
    #
    # fig.update_layout(scene_aspectmode='data',
    #                   scene=dict(
    #                       xaxis=dict(visible=False),
    #                       yaxis=dict(visible=False),
    #                       zaxis=dict(visible=False),
    #                     ),
    #                   scene_camera=camera_params
    #                   )
    # fig.show()
    ######### plot ball struct plot

    et = time.process_time()
    dur = et - st
    print('CPU Execution time in prepare_crave_FCC:', dur, 'seconds')


    if use_BCC == True:
        FCC_struct = SC2BCC(ball_struct, ratio)
    elif use_DIA == True:
        FCC_struct = SC2DIA(ball_struct, ratio)
    else:
        FCC_struct = SC2FCC(ball_struct, ratio)

    ######### plot FCC plot
    # FCC_HP = np.repeat(heritage_HP, 4)
    # FCC_plot = pd.DataFrame(FCC_struct, columns=['X', 'Y', 'Z'])
    # FCC_plot['HP'] = FCC_HP
    #
    # # fig = px.scatter_3d(x=FCC_plot['X'], y=FCC_plot['Y'], z=FCC_plot['Z'], color=FCC_plot['HP'], color_discrete_map=color_map)
    # fig = px.scatter_3d(x=FCC_plot['X'], y=FCC_plot['Y'], z=FCC_plot['Z'], color=FCC_plot['HP'],
    #                     opacity=0.2, color_discrete_map=color_map)
    # fig.add_trace(go.Scatter3d(x=X_axis[:, 0], y=X_axis[:, 1], z=X_axis[:, 2], mode='lines',
    #                            line=dict(color='rgba(0, 153, 0, 1)', width=10)))
    # fig.add_trace(go.Scatter3d(x=Y_axis[:, 0], y=Y_axis[:, 1], z=Y_axis[:, 2], mode='lines',
    #                            line=dict(color='rgba(255, 255, 0, 1)', width=10)))
    # fig.add_trace(go.Scatter3d(x=Z_axis[:, 0], y=Z_axis[:, 1], z=Z_axis[:, 2], mode='lines',
    #                            line=dict(color='rgba(204, 0, 0, 1)', width=10)))
    #
    #
    # fig.update_layout(scene_aspectmode='data',
    #                   scene=dict(
    #                       xaxis=dict(visible=False),
    #                       yaxis=dict(visible=False),
    #                       zaxis=dict(visible=False),
    #                     ),
    #                   scene_camera=camera_params
    #                   )
    # fig.show()
    ######### plot FCC plot
    Ro_FCC = np.dot(FCC_struct, R.T)
    ######### plot Ro plot


    # Ro_HP = np.repeat(heritage_HP, 4)
    # Ro_plot = pd.DataFrame(Ro_FCC, columns=['X', 'Y', 'Z'])
    # Ro_plot['HP'] = Ro_HP
    # fig = px.scatter_3d(x=Ro_plot['X'], y=Ro_plot['Y'], z=Ro_plot['Z'], color=Ro_plot['HP'], color_discrete_map=color_map, opacity=0.2)
    # fig.add_trace(go.Scatter3d(x=X_axis[:, 0], y=X_axis[:, 1], z=X_axis[:, 2], mode='lines',
    #                            line=dict(color='rgba(0, 153, 0, 1)', width=10)))
    # fig.add_trace(go.Scatter3d(x=Y_axis[:, 0], y=Y_axis[:, 1], z=Y_axis[:, 2], mode='lines',
    #                            line=dict(color='rgba(255, 255, 0, 1)', width=10)))
    # fig.add_trace(go.Scatter3d(x=Z_axis[:, 0], y=Z_axis[:, 1], z=Z_axis[:, 2], mode='lines',
    #                            line=dict(color='rgba(204, 0, 0, 1)', width=10)))
    #
    # fig.update_layout(scene_aspectmode='data',
    #                   scene=dict(
    #                       xaxis=dict(visible=False),
    #                       yaxis=dict(visible=False),
    #                       zaxis=dict(visible=False),
    #                     ),
    #                   scene_camera=camera_params
    #                   )
    # fig.show()
    ######### plot Ro plot4

    # Ro_FCC = np.dot(R, FCC_struct)
    Tr_FCC = move_to_Hcenter(Ro_FCC, Hcenter)
    del Ro_FCC
    del FCC_struct
    del cube_strcut
    del ball_struct
    # if draw == True:
    #     return Tr_FCC, FCC_HP

    return Tr_FCC   #Ball-shape but with orientation

def carve_FCC(Out, ratio, CI_R=100):  #circumscribed circle
    u_BCC = global_variable.use_bcc_lattice_struct   #####!!!
    u_DIA = global_variable.use_dia_lattice_struct

    Tr_FCC = prepare_carve_FCC(Out, ratio, use_BCC=u_BCC, use_DIA=u_DIA)

    st = time.process_time()
    tree_H = spatial.cKDTree(Out[['HX', 'HY', 'HZ']].to_numpy())
    tree_C = spatial.cKDTree(Tr_FCC)
    Crave_idx = tree_H.query_ball_tree(tree_C, r=CI_R)
    Crave_idx_unique = np.unique([item for items in Crave_idx for item in items])
    Crave_FCC = np.take(Tr_FCC, Crave_idx_unique, axis=0)

    et = time.process_time()
    dur = et - st
    print('CPU Execution time in crave_FCC:', dur, 'seconds')
    del Tr_FCC
    return Crave_FCC

def check_H_to_C(Out, FCC_data):
    # for_draw = check_SC2FCC(FCC_data)
    Out_sub = Out[['HX', 'HY', 'HZ']]
    Out_sub = Out_sub.rename(columns={'HX':'X', 'HY':'Y', 'HZ':'Z'})
    Out_sub['atom-refer'] = 0
    FCC_df = pd.DataFrame(FCC_data, columns=['X', 'Y', 'Z'])
    FCC_df['atom-refer'] = 1
    return pd.concat([FCC_df, Out_sub], ignore_index=True)

def carve_GB_M1(Margin, FCC_data):  ### np.rint(array)
    def find_FCC_margin_id(array):
        array_rint = np.rint(array)
        return array_rint

    st = time.process_time()
    Margin_sub = Margin[['HX', 'HY', 'HZ', 'margin-ID']]
    vfunc = np.vectorize(find_FCC_margin_id, signature='(n)->(m)')
    FCC_HXYZ_pre = vfunc(FCC_data)
    FCC_HXYZ = np.hstack((FCC_data, FCC_HXYZ_pre))

    FCC_df = pd.DataFrame(FCC_HXYZ, columns=['X', 'Y', 'Z', 'HX', 'HY', 'HZ'])
    FCC_margin_id = FCC_df.merge(Margin_sub, on=['HX', 'HY', 'HZ'], how='left')
    FCC_keep = FCC_margin_id[(FCC_margin_id['margin-ID'] == 2) | (FCC_margin_id['margin-ID'] == 0)]
    et = time.process_time()
    dur = et - st
    print('CPU Execution time in crave_GB:', dur, 'seconds')
    del FCC_df
    del FCC_margin_id
    del FCC_HXYZ
    del Margin_sub
    del FCC_HXYZ_pre

    return FCC_keep


# def crave_GB_M2(Inter_margin, FCC_data):  ### np.rint(array)
#     def vec_round_half(array):
#         return np.rint(array*2)/2
#
#     st = time.process_time()
#     Margin_sub = Inter_margin[['HX', 'HY', 'HZ', 'margin-ID']]
#
#     vfunc = np.vectorize(vec_round_half, signature='(n)->(m)')
#     FCC_HXYZ_half_pre = vfunc(FCC_data)
#     FCC_HXYZ_half = np.hstack((FCC_data, FCC_HXYZ_half_pre))
#
#     FCC_half_df = pd.DataFrame(FCC_HXYZ_half, columns=['X', 'Y', 'Z', 'HX', 'HY', 'HZ'])
#     FCC_half_margin_id = FCC_half_df.merge(Margin_sub, on=['HX', 'HY', 'HZ'], how='left')
#     FCC_keep = FCC_half_margin_id[(FCC_half_margin_id['margin-ID'] == 3)]
#
#     et = time.process_time()
#     dur = et - st
#     print('CPU Execution time in crave GB again:', dur, 'seconds')
#     return FCC_keep

def check_craved_GB(Inter_margin, FCC_GB, FCC_data):
    Inter_margin_sub = Inter_margin[['HX', 'HY', 'HZ', 'margin-ID']].rename(columns={'HX': 'X', 'HY': 'Y', 'HZ': 'Z', 'margin-ID':'atom-refer'})  ### H type
    FCC_GB_sub = FCC_GB[['X', 'Y', 'Z']]
    FCC_data_sub = pd.DataFrame(FCC_data, columns=['X', 'Y', 'Z'])
    FCC_delete = pd.concat([FCC_data_sub, FCC_GB_sub]).drop_duplicates(keep=False)
    FCC_GB_sub['atom-refer'] = -1   ### FCC
    FCC_delete['atom-refer'] = 4
    return pd.concat((Inter_margin_sub, FCC_delete))

def Process_pre(ID, cube_ratio):
    Out_ = global_variable.Frame_.from_ID_to_D(ID)
    Craved_FCC_ = carve_FCC(Out_, cube_ratio, CI_R=np.sqrt(2))
    del Out_
    return Craved_FCC_

def Process(ID, cube_ratio):
    Margin_ = global_variable.Frame_.find_grain_NN_with_out(ID)
    Craved_FCC_ = Process_pre(ID, cube_ratio)
    Craved_GB = carve_GB_M1(Margin_, Craved_FCC_)
    Craved_GB['grain-ID'] = ID
    del Margin_
    del Craved_FCC_
    return Craved_GB

def Process_Extend(ID, cube_ratio, unit_extend_ratio=global_variable.unit_extend_ratio):
    Out_ = global_variable.Frame_.find_grain_NN_with_out(ID)
    Extend_Out_ = global_variable.Frame_.get_extend_Out_(Out_, unit_extend_ratio)
    Extend_Out_ = global_variable.Frame_.renew_outer_margin(Extend_Out_)
    Extend_Out_XYZdf = Extend_Out_.rename(columns={"grain-ID": "ID"})#.round().astype(int)
    Craved_FCC_ = carve_FCC(Extend_Out_XYZdf, cube_ratio, CI_R=np.sqrt(2))
    del Extend_Out_XYZdf

    Craved_GB = carve_GB_M1(Extend_Out_, Craved_FCC_)
    Craved_GB['grain-ID'] = ID
    del Extend_Out_
    del Craved_FCC_
    return Craved_GB


if __name__ == "__main__":

    # import readDATA



    # print(Bass)
    # print(Struct)
    # print(Craved_FCC_)
    # print(Craved_FCC_)
    # import global_variable
    # def Process(ID, cube_ratio, center):
    #     Out_ = readDATA.from_ID_to_D(ID)
    #     Craved_FCC_ = crave_FCC(Out_, cube_ratio, center, CI_R=np.sqrt(2))
    #     Margin_ = readDATA.find_grain_NN_with_out(ID)
    #     Craved_GB = crave_GB_M1(Margin_, Craved_FCC_)
    #     Craved_GB['grain-ID'] = ID
    #     return Craved_GB

    # G1 = Process(360, 1, center=[ 44.69068879, 149.23395688, 313.75726518])
    # G2 = Process(523, 1, center=[ 40.77112809, 165.5579445,  319.64339332])#
    # G_ = pd.concat([G1, G2])
    #
    # with open(global_variable.raw_dir, 'a') as f:
    #     G_.to_csv(f, index=False, header=False)

    ID_ = 178
    #
    # Out_ = global_variable.Frame_.from_ID_to_D(ID_)
    # Out_df = pd.DataFrame(Out_, columns=['HX', 'HY', 'HZ'])

    # Tr_FCC = prepare_carve_FCC(Out_, 10, random=True, draw=False, use_BCC=False)

    # Tr_FCC_df = pd.DataFrame(Tr_FCC, columns=['HX', 'HY', 'HZ'])
    # Tr_FCC_df_sample = Tr_FCC_df.sample(frac=0.01)
    # fig = px.scatter_3d(x=ball_struct_df['HX'], y=ball_struct_df['HY'], z=ball_struct_df['HZ'])
    # fig.show()

    # fig = px.scatter_3d(x=Tr_FCC_df_sample['HX'], y=Tr_FCC_df_sample['HY'], z=Tr_FCC_df_sample['HZ'])
    # fig.show()


    # Grain = Process(1234,)
    print(Grain)
    # #
    fig = px.scatter_3d(x=Grain['X'], y=Grain['Y'], z=Grain['Z'])
    fig.show()
    #
    # Out_ = global_variable.Frame_.find_grain_NN_with_out(ID_, with_diagonal=False)
    # Out_BCC = Out_[Out_['grain-ID'] == ID_]
    #
    # grain_check = check_H_to_C(Out_BCC, Grain)


    # def check_lattice_constant_simple(FCC_Construct, used_ratio):
    #     check_lattice_constant_pool = FCC_Construct[FCC_Construct['margin-ID'] == 0]
    #     # print(check_lattice_constant_pool)
    #     check_for_2_idx = np.random.randint(len(check_lattice_constant_pool), size=2)
    #     check_lattice_selected = check_lattice_constant_pool.iloc[check_for_2_idx]
    #
    #     check_lattice_selected_XYZ = check_lattice_selected[['X', 'Y', 'Z']].to_numpy()
    #     check_lattice_constant_pool_XYZ = check_lattice_constant_pool[['X', 'Y', 'Z']].to_numpy()
    #     tree = spatial.cKDTree(check_lattice_constant_pool_XYZ)
    #     upper_bound = used_ratio * 0.87  # (FCC)np.sqrt(2)/2 = 0.75, # (BCC)np.sqrt(3)/2 = 0.87
    #     _, index = tree.query(check_lattice_selected_XYZ, k=8, distance_upper_bound=upper_bound)
    #
    #     boundary_index = len(check_lattice_constant_pool)
    #     for each_row_idx in range(len(index)):
    #         each_row = index[each_row_idx]
    #         index_row = np.delete(each_row, np.argwhere(each_row == boundary_index))
    #         selected_NN = np.take(check_lattice_constant_pool_XYZ, index_row, axis=0)
    #         select_point = check_lattice_selected_XYZ[each_row_idx][np.newaxis, :]
    #         all_distance = cdist(select_point, selected_NN)
    #         all_distance = all_distance[all_distance != 0]
    #
    #         print(all_distance)
    #     return


    # check_lattice_constant_simple(Grain, 1)


    # print(Grain_check)
    # print(len(Grain))
    # Grain = Process(111, 1.9)
    # print(len(Grain))

    # Out_ = global_variable.Frame_.from_ID_to_D(111)
    # print(len(Out_))
    # print(len(SC2FCC(Out_,1)))


    # print(Out_)
    # Tr_FCC = prepare_carve_FCC(Out_, 2, random=True, draw=False, use_BCC=False)
    # print(len(Tr_FCC))



# Grain = Process(111, 1.8)
    # print(len(Grain))
    # Grain = Process(111, 1.7)
    # print(len(Grain))
    # Grain = Process(111, 1.6)
    # print(len(Grain))
    # Grain = Process(111, 1.5)
    # print(len(Grain))
    # Grain = Process(111, 1.4)
    # print(len(Grain))
    # Grain = Process(111, 1.3)
    # print(len(Grain))
    # Grain = Process(111, 1.2)
    # print(len(Grain))
    # Grain = Process(111, 1.1)
    # print(len(Grain))
    # Grain = Process(111, 1.0)
    # print(len(Grain))
    # Grain = Process(111, 1.5)
    # print(len(Grain))

    # ID = 2234
    # Out_18 = global_variable.Frame_.find_grain_NN_with_out(ID)
    # print(len(Out_18))
    # print(Out_18['HX'].max())
    # print(Out_18['HX'].min())
    # print(Out_18['HY'].max())
    # print(Out_18['HY'].min())
    # print(Out_18['HZ'].max())
    # print(Out_18['HZ'].min())
    # unique_ID = np.unique(Out_18['grain-ID'].to_numpy())
    # print(unique_ID)

    # Extend_Out_18 = global_variable.Frame_.get_extend_Out_(Out_18, global_variable.unit_extend_ratio)
    # Extend_Out_18 = global_variable.Frame_.renew_outer_margin(Extend_Out_18)
    # #
    # Extend_Out_margin_18 = Extend_Out_18[Extend_Out_18['margin-ID'] != 0]
    # #
    # final_18 = Process_Extend(ID, 1.5, unit_extend_ratio=global_variable.unit_extend_ratio)
    #
    # print(final_18['X'].min())
    # print(final_18['X'].max())
    # print(final_18['Y'].min())
    # print(final_18['Y'].max())
    # print(final_18['Z'].min())
    # print(final_18['Z'].max())

    # ID = 15219
    # Out_126 = global_variable.Frame_.find_grain_NN_with_out(ID)
    # Extend_Out_126 = global_variable.Frame_.get_extend_Out_(Out_126, global_variable.unit_extend_ratio)
    # Extend_Out_126 = global_variable.Frame_.renew_outer_margin(Extend_Out_126)
    #
    # Extend_Out_margin_126 = Extend_Out_126[Extend_Out_126['margin-ID'] != 0]
    #
    # Extend_both = pd.concat([Extend_Out_margin_126, Extend_Out_margin_18])
    # print(np.unique(Extend_Out_['grain-ID']))
    # print(Extend_Out_)

    # Out_ = global_variable.Frame_.find_grain_NN_with_out(111, with_diagonal=False)
    # Out_FCC = Out_[Out_['grain-ID'] != 111]

    # new_row = {'HX': 1000, 'HY': 1000, 'HZ': 1000, 'margin-ID': 1, 'grain-ID': 111}

    # Append the new row to the DataFrame
    # Out_FCC = pd.concat([
    #             Out_FCC,
    #             pd.DataFrame([[75, 307, 20, 1, 111]], columns=['HX', 'HY', 'HZ', 'margin-ID', 'grain-ID'])]
    #        ).reset_index(drop=True)


    # Craved_FCC_ = carve_FCC(Out_, 1.5, CI_R=np.sqrt(2))
    # print(Craved_FCC_)
    # print(Out_)
    # grain_df = pd.DataFrame(Out_, columns=['X', 'Y', 'Z', 'ID'])
    # del Out_
    # return Craved_FCC_


        # del FCC_Construct_

    # camera_params = dict(
    #     up=dict(x=0, y=0, z=1),
    #     center=dict(x=0, y=0, z=0),
    #     eye=dict(x=1.25, y=0, z=0)
    # )
    #


    # #
    # fig_draw = px.scatter_3d(Extend_Out_margin_18, x='HX', y='HY', z='HZ', color='margin-ID', opacity=0.5)
    #
    # final_draw = px.scatter_3d(final_18, x='X', y='Y', z='Z',color_discrete_sequence=['blueviolet'])
    #
    # fig = go.Figure(
    #     data=final_draw.data) # fig_draw.data +
    #
    # fig.update_layout(scene_aspectmode='data',
    #                   scene=dict(
    #                       xaxis=dict(visible=True),
    #                       yaxis=dict(visible=True),
    #                       zaxis=dict(visible=True),
    #                     ),
    #                   # scene_camera=camera_params
    #                   )
    #
    #
    # fig.show()



