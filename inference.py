# Input modules
import os
import sys
import subprocess
import argparse
import time
import json
from PIL import Image
from utils import BlenderAlchemy_run, tree_dim_parse

task_instance_count_dict = {
    'geometry': 50,
    'material': 40,
    'blendshape': 75,
    'placement': 40,
    'lighting': 40
}

def VLMSystem_run(blender_file_path, start_script, start_render, goal_render, blender_render_script_path, task_instance_id, task, infinigen_installation_path):
    '''
    API for user-implemented VLM-system. With only a VLM, rather than a system of VLMs, the user is encouraged to use our implementation of BlenderAlchemy. Check out guide on readme.md. 
    
    Generation and potentially selection process of the VLM system.

    Inputs:
        blender_file_path: file path to the .blend base file
        start_file_path: file path to the start.py, the script for start scene
        start_render_path: dir path to the rendered images of start scene
        goal_render: dir path to the rendered images of goal scene
        blender_render_script_path: file path to the render script of blender scene
        task: name of the task, like `geometry`, `placement`
        task_instance_id: f'{task}{i}', like `placement1`, `geometry2`
        infinigen_installation_path: file/dir path to infinigen blender executable file for background rendering

    Outputs:
        proposal_edits_paths: a list of file paths to proposal scripts from the VLM system 
        selected_edit_path[optional]: if applicable, the file path to the VLM-system-selected proposal script
    '''

    proposal_edits_paths = None
    proposal_renders_paths = None
    selected_edit_path = None
    selected_render_path = None

    return proposal_edits_paths, proposal_renders_paths, selected_edit_path, selected_render_path

if __name__=='__main__':
    '''
    Input args are listed here. 
    '''
    parser = argparse.ArgumentParser(description='Image-based program edits')

    parser.add_argument('--task', 
        type=str, default="test", 
        help="`all`, `test`, `subset`, or comma-separated list of the following: `material`, `geometry`, `blendshape`, `placement`,`lighting`"
    )

    parser.add_argument('--info_saving_dir_path', 
        type=str, default="info_saved", 
        help='''Directory that intermediate inference metadata, such as path to all edit scripts and images and the final output from VLM, is saved. 
       The json file that pass all BlenderGym inference metadata to evaluation(calculation of performance scores) is saved here. By default, this is info_saved/'''
    )

    parser.add_argument('--blender_render_script_path', 
        type=str, default=f"{os.path.abspath(os.path.join('bench_data', 'pipeline_render_script.py'))}", 
        help="The Blender render script. By default, it's bench_data/pipeline_render_script.py, which uses two views for VLM generation/verification."
    )

    parser.add_argument('--infinigen_installation_path', 
        type=str, default=f"{os.path.abspath('infinigen/blender/blender')}", 
        help="The installation path of blender executable file. It's `infinigen/blender/blender` by default."
    )

    parser.add_argument('--custom_vlm_system', 
        action='store_true', 
        help='''Whether change our VLM setup. If you want to change our VLM system (i.e. rewire/delete our generator-verifier structure)
        If you choose to use your custom_vlm_system, please implement the function VLMSystem_run().'''
    )

    parser.add_argument('--generator_type', 
        type=str, default=None, 
        help="model_id of VLM generator. Note this is the specific id listed in Supported Models or named by you."
    )
    
    parser.add_argument('--verifier_type', 
        type=str, default=None, 
        help="model_id of VLM verifier. Note this is the specific id listed in Supported Models or named by you."
    )
    
    parser.add_argument('--tree_dims', 
        type=str, default='3x4', 
        help="Tree dimension for generation-verification tree. We set the default to 3x4, aligned with BlenderGym configuration."
    )

    # parse, save, and validate the args
    args = parser.parse_args()
    tasks = args.task.strip().split(',')
    info_saving_dir_path = args.info_saving_dir_path
    blender_render_script_path = args.blender_render_script_path
    infinigen_installation_path = args.infinigen_installation_path
    tree_dims = tree_dim_parse(args.tree_dims)

    # infinigen_installation_path = 
    custom_vlm_system = args.custom_vlm_system
    generator_type = args.generator_type
    verifier_type = args.verifier_type

    if len(tasks) == 0:
        raise ValueError(f'Invalid input for --task: no task input detected.')
    
    for task in tasks:
        if task not in ('all', 'test', 'subset') and task not in task_instance_count_dict.keys():
            raise ValueError(f'Invalid input for --task: {task} is not a invalid input for "--task."')

    if not os.path.isfile(blender_render_script_path):
        raise ValueError(f'Invalid input for blender_render_script_path: {blender_render_script_path}')

    if not os.path.exists(infinigen_installation_path):
        raise ValueError(f'Invalid input for infinigen_installation_path: {infinigen_installation_path}')
    infinigen_installation_path = os.path.abspath(infinigen_installation_path)

    os.makedirs(info_saving_dir_path, exist_ok=True)
    info_saving_json_path = os.path.join(info_saving_dir_path, f'intermediate_metadata_{time.strftime("%m-%d-%H-%M-%S")}.json')

    # Load in instance dir paths
    if 'all' in tasks:
        task_instance_dir_paths = {task:[os.path.join('bench_data', f'{task}{i}') for i in range(1, task_instance_count_dict[task]+1)] for task in task_instance_count_dict.keys()}
    elif 'test' in tasks:
        task_instance_dir_paths = {task:[os.path.join('bench_data', f'{task}{i}') for i in range(1, 4)] for task in task_instance_count_dict.keys()}
    elif 'subset' in tasks:
        task_instance_dir_paths = {task:[os.path.join('bench_data', f'{task}{i}') for i in range(1, 11)] for task in task_instance_count_dict.keys()}
    else:
        task_instance_dir_paths = {task:[os.path.join('bench_data', f'{task}{i}') for i in range(1, task_instance_count_dict[task]+1)] for task in tasks}
    # eg in instance_dir_paths: bench_data/geometry1


    print(f'task_instance_dir_paths: {task_instance_dir_paths}')

    starter_time = time.strftime("%m-%d-%H-%M-%S")
    if not args.custom_vlm_system:
        generation_results = {"output_dir_name":f"outputs_{starter_time}", 'generator_type':generator_type, 'evaluator_type': evaluator_type, 'tree_dims': tree_dims}
    else:
        generation_results = {"output_dir_name":f"outputs_{starter_time}"}


    # Run the pipeline on each instance dir
    for task, instance_dir_paths in task_instance_dir_paths.items():
        # Register the task to the results-saving dict
        generation_results[task] = {}

        for instance_dir_path in instance_dir_paths:
            # Register the task instance to the results-saving dict
            task_instance_id = os.path.basename(instance_dir_path)
            generation_results[task][task_instance_id] = {}

            # Define input for the VLM system
            instance_dir_path = os.path.abspath(instance_dir_path)
            blender_file_path = os.path.join(instance_dir_path, 'blender_file.blend')
            start_file_path = os.path.join(instance_dir_path, 'start.py')
            start_render_path = os.path.join(instance_dir_path, 'renders/start')
            goal_file_path = os.path.join(instance_dir_path, 'goal.py')
            goal_render_path = os.path.join(instance_dir_path, 'renders/goal')

            # Call the VLM system
            if not custom_vlm_system:
                if not generator_type or not evaluator_type:
                    raise ValueError("For VLM-only usage, please indicate both generator and evaluator model.")
                try:
                    proposal_edits_paths, proposal_renders_paths, selected_edit_path, selected_render_path = BlenderAlchemy_run(blender_file_path, start_file_path, start_render_path, goal_render_path, blender_render_script_path, task_instance_id, task, infinigen_installation_path, generator_type, evaluator_type, starter_time=starter_time, tree_dims=tree_dims)    
                except:
                    continue
            else:
                proposal_edits_paths, proposal_renders_paths, selected_edit_path, selected_render_path = VLMSystem_run(blender_file_path, start_file_path, start_render_path, goal_render_path, blender_render_script_path, task_instance_id, task, infinigen_installation_path)    

            # DEBUG:
            print(f'proposal_edits_paths: {proposal_edits_paths}')
            print(f'proposal_renders_paths: {proposal_renders_paths}')
            print(f'selected_edit_path: {selected_edit_path}')
            print(f'selected_render_path: {selected_render_path}')

            # Save the results for the VLM system
            generation_results[task][task_instance_id]['instance_dir_path'] = instance_dir_path
            generation_results[task][task_instance_id]['blender_file_path'] = blender_file_path
            generation_results[task][task_instance_id]['start_script_path'] = start_file_path
            generation_results[task][task_instance_id]['goal_script_path'] = goal_file_path
            generation_results[task][task_instance_id]['proposal_edits_paths'] = proposal_edits_paths
            generation_results[task][task_instance_id]['proposal_renders_paths'] = proposal_renders_paths
            generation_results[task][task_instance_id]['selected_edit_path'] = selected_edit_path
            generation_results[task][task_instance_id]['selected_render_path'] = selected_render_path

            with open(info_saving_json_path, 'w') as file:
                json.dump(generation_results, file, indent=4)











