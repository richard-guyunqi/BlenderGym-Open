import os
import subprocess
from tqdm import tqdm
from huggingface_hub import hf_hub_download

env = os.environ.copy()

# Step 1: download the blendergym data zip file
if not os.path.isfile('bench_data.zip') and not os.path.isdir('bench_data'):
    command = '''
    wget https://huggingface.co/datasets/richard-guyunqi/BG_bench_data/resolve/main/bench_data.zip
    '''
    subprocess.run(command, env=env, shell=True)


# Step 2: Unzip the dataset and organize the files
if not os.path.isdir('bench_data'):
    command = '''
        unzip bench_data.zip
    '''
    subprocess.run(command, env=env, shell=True)

    # Step 2: for each blender_file, copy it to the correct place
    blender_files_dir = "bench_data/blender_files"

    for blender_file_name in tqdm(os.listdir(blender_files_dir)):
        blender_file_path = os.path.join(blender_files_dir, blender_file_name)
        task, start, end = blender_file_name.split('.')[0].split('_')

        start = int(start)
        end = int(end)


        for i in range(start, end +1):
            command = f"cp {blender_file_path} bench_data/{task}{i}/blender_file.blend"
            print(command)

            subprocess.run(command, env=env, shell=True)












# # Step 4: 




