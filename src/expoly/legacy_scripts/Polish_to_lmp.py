import pandas as pd
import numpy as np
np.set_printoptions(threshold=100)
pd.options.mode.chained_assignment = None  # default='warn'

import grid
import global_variable
from jinja2 import Environment, FileSystemLoader
from scipy import spatial
from scipy.spatial.distance import cdist
import os

from ovito.io import import_file, export_file
from ovito.data import CutoffNeighborFinder
from ovito.modifiers import DeleteSelectedModifier
import subprocess

import plotly.express as px


###### all variable and directory control in global variable

def check_lattice_constant_simple(FCC_Construct, used_ratio):
    check_lattice_constant_pool = FCC_Construct[FCC_Construct['margin-ID'] == 4]
    check_for_2_idx = np.random.randint(len(check_lattice_constant_pool), size=2)
    check_lattice_selected = check_lattice_constant_pool.iloc[check_for_2_idx]

    check_lattice_selected_XYZ = check_lattice_selected[['X', 'Y', 'Z']].to_numpy()
    check_lattice_constant_pool_XYZ = check_lattice_constant_pool[['X', 'Y', 'Z']].to_numpy()
    tree = spatial.cKDTree(check_lattice_constant_pool_XYZ)
    upper_bound = used_ratio * 0.75  # np.sqrt(2)/2
    _, index = tree.query(check_lattice_selected_XYZ, k=13, distance_upper_bound=upper_bound)

    boundary_index = len(check_lattice_constant_pool)
    for each_row_idx in range(len(index)):
        each_row = index[each_row_idx]
        index_row = np.delete(each_row, np.argwhere(each_row == boundary_index))
        selected_NN = np.take(check_lattice_constant_pool_XYZ, index_row, axis=0)
        select_point = check_lattice_selected_XYZ[each_row_idx][np.newaxis, :]
        all_distance = cdist(select_point, selected_NN)
        all_distance = all_distance[all_distance != 0]

        print(all_distance)
    return

def ask_for_initial_in_dir():
    if os.path.exists(global_variable.in_dir):
        print("Exist lammps input file %s, clean or not(Y/n)"%global_variable.in_dir)
        clean = input()
        if clean == 'Y':
            os.remove(global_variable.in_dir)
            print("cleaned %s"%global_variable.in_dir)
            return True
        elif clean == 'n':
            return False
        else:
            print("Wrong input")
            return ask_for_initial_in_dir()
    else:
        return True

def auto_initial_in_dir():
    if os.path.exists(global_variable.in_dir):
        os.remove(global_variable.in_dir)
        print("cleaned %s"%global_variable.in_dir)
    if os.path.exists(global_variable.PSC_dir):
        os.remove(global_variable.PSC_dir)
        print("cleaned %s" % global_variable.PSC_dir)
    if os.path.exists(global_variable.Overlap_dir):
        os.remove(global_variable.Overlap_dir)
        print("cleaned %s" % global_variable.Overlap_dir)
    if os.path.exists(global_variable.ID_dir):
        os.remove(global_variable.ID_dir)
        print("cleaned %s" % global_variable.ID_dir)
    if os.path.exists(global_variable.lmp_dir):
        os.remove(global_variable.lmp_dir)
        print("cleaned %s" % global_variable.lmp_dir)
    else:
        print("")
    return

def ask_for_initial_raw_dir():
    if os.path.exists(global_variable.raw_dir):
        print("Exist raw data directory %s, clean or not(Y/n)"%global_variable.raw_dir)
        clean = input()
        if clean == 'Y':
            os.remove(global_variable.raw_dir)
            print("cleaned %s"%global_variable.raw_dir)
            return True
        elif clean == 'n':
            return False
        else:
            print("Wrong input")
            return ask_for_initial_raw_dir()
    else:
        return True

def polish_to_lmp():
    answer = auto_initial_in_dir()
    if answer == False:
        print("file not changed")
        return
    else:
        use_cube_ratio = global_variable.use_cube_ratio
        # quater_lattice_constant = global_variable.lattice_constant * 0.25   #why added?
        scan_ratio = global_variable.scan_ratio

        this_file = pd.read_csv(global_variable.raw_dir, header=None)
        this_file.columns = ['X', 'Y', 'Z', 'HX', 'HY', 'HZ', 'margin-ID', 'grain-ID']
        if global_variable.real_extent == False:
            HX_range = global_variable.HX_range
            HY_range = global_variable.HY_range
            HZ_range = global_variable.HZ_range
        elif global_variable.real_extent == True:
            HX_range = np.array(global_variable.HX_range) * global_variable.unit_extend_ratio
            HY_range = np.array(global_variable.HY_range) * global_variable.unit_extend_ratio
            HZ_range = np.array(global_variable.HZ_range) * global_variable.unit_extend_ratio
        else:
            print("wrong real_extent value")
            HX_range = global_variable.HX_range
            HY_range = global_variable.HY_range
            HZ_range = global_variable.HZ_range

        cut_area = this_file[(this_file['HX'] >= HX_range[0]) & (this_file['HX'] <= HX_range[1]) &
                             (this_file['HY'] >= HY_range[0]) & (this_file['HY'] <= HY_range[1]) &
                             (this_file['HZ'] >= HZ_range[0]) & (this_file['HZ'] <= HZ_range[1])]

        DATA_ = cut_area.copy()
        DATA_["X"] = (DATA_["X"] - HX_range[0]) * scan_ratio
        DATA_["Y"] = (DATA_["Y"] - HY_range[0]) * scan_ratio
        DATA_["Z"] = (DATA_["Z"] - HZ_range[0]) * scan_ratio
        atom_num = DATA_.shape[0]
        grain_num = len(np.unique(DATA_['grain-ID']))

        X_low = DATA_['X'].min()
        X_high = DATA_['X'].max()
        Y_low = DATA_['Y'].min()
        Y_high = DATA_['Y'].max()
        Z_low = DATA_['Z'].min()
        Z_high = DATA_['Z'].max()

        environment = Environment(loader=FileSystemLoader(global_variable.EXPoly_dir))
        template = environment.get_template("template_polish.txt")  # always name template.txt
        content = template.render(
            grain_num=grain_num,
            current_frame=global_variable.current_frame,
            cube_ratio=use_cube_ratio,
            HX_low=HX_range[0],
            HX_high=HX_range[1],
            HY_low=HY_range[0],
            HY_high=HY_range[1],
            HZ_low=HZ_range[0],
            HZ_high=HZ_range[1],
            atom_num=atom_num,
            X_low=X_low,
            X_high=X_high,
            Y_low=Y_low,
            Y_high=Y_high,
            Z_low=Z_low,
            Z_high=Z_high,
            force_return='\n'
        )
        with open(global_variable.in_dir, mode="w", encoding="utf-8") as message:
            message.write(content)
            print(f"... wrote %s title" % global_variable.in_dir)

        DATA_ = DATA_.reset_index(drop=True)
        DATA_['atom-type'] = 1
        DATA_['atom-ID'] = DATA_.index + 1
        write_subdata = DATA_[['atom-ID', 'atom-type', 'X', 'Y', 'Z']]
        write_subdata.to_csv(global_variable.in_dir, mode='a', sep=' ', index=False, header=False)

        write_ID = DATA_[['atom-ID', 'X', 'Y', 'Z', 'margin-ID', 'grain-ID']]
        write_ID.to_csv(global_variable.ID_dir, mode='a', sep=' ', index=False, header=False)

        print(f"... wrote %s Atoms" % global_variable.in_dir)
        del DATA_, this_file, cut_area
        return

def inlay_to_lmp():
    auto_initial_in_dir()

    these_FCC_grains_pre = pd.read_csv(global_variable.raw_dir, header=None)
    these_FCC_grains_pre.columns = ['X', 'Y', 'Z', 'HX', 'HY', 'HZ', 'margin-ID', 'grain-ID']

    # use defined cube H structure
    HX_range = global_variable.HX_range
    HY_range = global_variable.HY_range
    HZ_range = global_variable.HZ_range

    these_FCC_grains = these_FCC_grains_pre[(these_FCC_grains_pre['HX'] >= HX_range[0]) & (these_FCC_grains_pre['HX'] <= HX_range[1]) &
                         (these_FCC_grains_pre['HY'] >= HY_range[0]) & (these_FCC_grains_pre['HY'] <= HY_range[1]) &
                         (these_FCC_grains_pre['HZ'] >= HZ_range[0]) & (these_FCC_grains_pre['HZ'] <= HZ_range[1])]



    Base_FCC = grid.Create_Base_FCC(ratio=global_variable.use_cube_ratio)

    if global_variable.with_base == True:
        inlay_structure = pd.concat([these_FCC_grains, Base_FCC])
    elif global_variable.with_base == False:
        inlay_structure = these_FCC_grains
    else:
        print("wrong with_base and use base")
        inlay_structure = pd.concat([these_FCC_grains, Base_FCC])


    scan_ratio = global_variable.scan_ratio
    inlay_structure["X"] = (inlay_structure["X"] - inlay_structure["X"].min()) * scan_ratio
    inlay_structure["Y"] = (inlay_structure["Y"] - inlay_structure["Y"].min()) * scan_ratio
    inlay_structure["Z"] = (inlay_structure["Z"] - inlay_structure["Z"].min()) * scan_ratio

    # half_lattice_constant = global_variable.lattice_constant * 0.125

    grain_num = len(global_variable.grain_ID_list)
    atom_num = inlay_structure.shape[0]
    # X_high = inlay_structure["X"].max() # + half_lattice_constant
    # Y_high = inlay_structure["Y"].max() # + half_lattice_constant
    # Z_high = inlay_structure["Z"].max() # + half_lattice_constant

    environment = Environment(loader=FileSystemLoader(global_variable.EXPoly_dir))
    template = environment.get_template("template_inlay.txt")  # always name template.txt
    content = template.render(
        grain_num=grain_num,
        current_frame=global_variable.current_frame,
        cube_ratio=global_variable.use_cube_ratio,
        HX_low=HX_range[0],  # inlay_structure["HX"].min(),
        HX_high=HX_range[1],  # inlay_structure["HX"].max(),
        HY_low=HY_range[0],  # inlay_structure["HY"].min(),
        HY_high=HY_range[1],  # inlay_structure["HY"].max(),
        HZ_low=HZ_range[0],  # inlay_structure["HZ"].min(),
        HZ_high=HZ_range[1],  # inlay_structure["HZ"].max(),
        atom_num=atom_num,
        X_low=inlay_structure["X"].min(),
        X_high=inlay_structure["X"].max(),
        Y_low=inlay_structure["Y"].min(),
        Y_high=inlay_structure["Y"].max(),
        Z_low=inlay_structure["Z"].min(),
        Z_high=inlay_structure["Z"].max(),
        force_return='\n'
    )
    with open(global_variable.in_dir, mode="w", encoding="utf-8") as message:
        message.write(content)
        print(f"... wrote %s title" % global_variable.in_dir)

    DATA_ = inlay_structure.reset_index(drop=True)
    DATA_['atom-type'] = 1
    DATA_['atom-ID'] = DATA_.index + 1
    write_subdata = DATA_[['atom-ID', 'atom-type', 'X', 'Y', 'Z']]
    write_subdata.to_csv(global_variable.in_dir, mode='a', sep=' ', index=False, header=False)

    write_ID = DATA_[['margin-ID', 'grain-ID']]
    write_ID.to_csv(global_variable.ID_dir, mode='a', sep=' ', index=False, header=False)
    print(f"... wrote %s Atoms" % global_variable.in_dir)
    del DATA_, these_FCC_grains, inlay_structure
    return

def Ovito_delete_overlap():
    pipeline = import_file(global_variable.in_dir)
    overlap_distance = 1.6
    def NonOverlapAtom(frame, data):
        # Show this text in the status bar while the modifier function executes
        yield "Selecting overlapping particles"
        # Create 'Selection' output particle property
        selection = data.particles_.create_property('Selection')
        # Prepare neighbor finder
        finder = CutoffNeighborFinder(overlap_distance, data)
        # Iterate over all particles
        for index in range(data.particles.count):

            # Update progress display in the status bar
            yield (index / data.particles.count)

            # Iterate over all nearby particles around the current center particle
            for neigh in finder.find(index):

                # Once we find a neighbor which hasn't been marked yet,
                # mark the current center particle. This test is to ensure that we
                # always select only one of the particles in a close pair.
                if selection[neigh.index] == 0:
                    selection[index] = 1
                    break

    pipeline.modifiers.append(NonOverlapAtom)
    data = pipeline.compute()
    data_select = data.particles['Selection'][...]
    np.savetxt(global_variable.Overlap_dir, data_select,
               delimiter=",")

    pipeline.modifiers.append(DeleteSelectedModifier(operate_on={'particles'}))
    export_file(pipeline, global_variable.PSC_dir, "lammps/data")
    return

def merge_to_one():
    with open(global_variable.in_dir) as file:
        annote = file.readlines()[0:2]

    with open(global_variable.PSC_dir) as file:
        title = file.readlines()[1:15]

    with open(global_variable.lmp_dir, mode='a') as f:
        f.writelines(annote)
        f.write('\n')
        f.writelines(title)

    DATA_ = pd.read_csv(global_variable.PSC_dir, sep=" ", header=None, skiprows=15, engine='c')
    DATA_[0] = DATA_.index + 1

    DATA_.to_csv(global_variable.lmp_dir, mode='a', sep=' ', index=False, header=False)
    os.unlink(global_variable.in_dir)
    os.unlink(global_variable.PSC_dir)
    return

# def automatic_process():
#     if global_variable.series == 'inlay':
#         inlay_to_lmp()
#         Ovito_delete_overlap()
#         merge_to_one()
#         print("finish generate lammps in file")
#
#     elif global_variable.series == 'polish':
#         polish_to_lmp()
#         Ovito_delete_overlap()
#         merge_to_one()
#         print("finish generate lammps in file")
#     else:
#         print("something wrong")
#     return

def automatic_process():

    polish_to_lmp()
    Ovito_delete_overlap()
    merge_to_one()
    print("finish generate lammps in file")
    return


if __name__ == '__main__':

    subprocess.run(['python', 'Multiprocessing.py'])
    print("Finish")

    automatic_process()









