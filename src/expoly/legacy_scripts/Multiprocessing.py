import global_variable
import grid
import multiprocessing

def process_and_write(ID):
    with open(global_variable.raw_dir, 'a') as f:
        if global_variable.real_extent == False:
            FCC_Construct_ = grid.Process(ID, cube_ratio=global_variable.use_cube_ratio)
            FCC_Construct_.to_csv(f, index=False, header=False)
            del FCC_Construct_
        elif global_variable.real_extent == True:
            FCC_Construct_ = grid.Process_Extend(ID, cube_ratio=global_variable.use_cube_ratio, unit_extend_ratio=global_variable.unit_extend_ratio)
            FCC_Construct_.to_csv(f, index=False, header=False)
            del FCC_Construct_
        else:
            print("Wrong real_extent")
    return

    ######## multiprocessing
if __name__ == '__main__':
    grain_ID_list = global_variable.grain_ID_list

    # global_variable.raw_dir.unlink(missing_ok=True)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:       # multiprocessing.cpu_count()
        pool.map(process_and_write, grain_ID_list)