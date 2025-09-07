# import Inlay_to_lmp
import Polish_to_lmp
import subprocess
import global_variable


# answer = Inlay_to_lmp.ask_for_initial_raw_dir()
# if answer == True:
#     subprocess.run(['python', 'Inlay_to_lmp.py'])
#
# else:
#     print("Nothing changed! or/and check your raw directory")

answer = Polish_to_lmp.ask_for_initial_raw_dir()
if answer == True:
    subprocess.run(['python', 'Polish_to_lmp.py'])
elif answer == False:
    Polish_to_lmp.automatic_process()
else:
    print("Nothing changed! or/and check your raw directory")
#
# elif global_variable.series == 'inlay':
#     answer = Inlay_to_lmp.ask_for_initial_raw_dir()
#     if answer == True:
#         subprocess.run(['python', 'Inlay_to_lmp.py'])
#     else:
#         print("Nothing changed! or/and check your raw directory")
#
# else:
#     print("check input in global variable series")