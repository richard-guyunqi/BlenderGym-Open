# TODO: Implement Chamfer for 3D tasks

# Input modules
import os
import sys
import subprocess
import argparse
import time
import json
from PIL import Image
from utils import photometric_loss, img2img_clip_similarity, blender_step, clip_similarity
from tqdm import tqdm

task_instance_count_dict = {
    'geometry': 45,
    'material': 40,
    'blendshape': 75,
    'placement': 40,
    'lighting': 40
}

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Image-based program edits')

    parser.add_argument('--inference_metadata_saved_path', 
        type=str, 
        help="Path to the inference metadata in json format (paths of proposal edit scripts, winner information, etc.)"
    )

    parser.add_argument('--eval_render_save_dir', 
        type=str, default=None, 
        help="The directory that all evaluation renders will be saved to.."
    )

    parser.add_argument('--infinigen_installation_path', 
        type=str, default=f"{os.path.abspath('infinigen/blender/blender')}", 
        help="The installation path of blender executable file. It's `infinigen/blender/blender` by default."
    )

    # parse, save, and validate the args
    args = parser.parse_args()
    inference_metadata_saved_path = args.inference_metadata_saved_path
    eval_render_save_dir = args.eval_render_save_dir
    infinigen_installation_path = args.infinigen_installation_path

    blender_render_script_path = "bench_data/all_render_script.py"

    if not os.path.isfile(inference_metadata_saved_path):
        raise ValueError(f'Invalid input for --inference_metadata_saved_path: {inference_metadata_saved_path}.')

    # Load in the data from pipeline inference
    with open(inference_metadata_saved_path, 'r') as file:
        inference_metadata = json.load(file)

    # Derive name for eval_render_save_dir
    if not eval_render_save_dir:
        eval_render_save_dir = f"eval_renders/{inference_metadata['output_dir_name']}"
    os.makedirs(eval_render_save_dir, exist_ok=True)

    tasks = inference_metadata.keys()

    # Create eval renders for all instances

    scores_across_tasks = {}
    intermediates = {}

    for task in tasks:
        if task not in task_instance_count_dict.keys():
            continue
        
        # Iterate through each instance
        scores_across_instances = {'best_n_clip':[], 'selected_n_clip':[], 'best_pl':[], 'selected_pl':[]}

        for task_instance, instance_info in inference_metadata[task].items():
            task_instance_dir = os.path.join(eval_render_save_dir, task_instance)
            os.makedirs(task_instance_dir, exist_ok=True)

            # Store local scores: score for each executable render
            task_instance_scores = {}

            try:
                # Iterate through each proposal_renders_path
                blender_file_path = instance_info['blender_file_path']
                start_file_path = instance_info['start_script_path']
                goal_file_path = instance_info['goal_script_path']
            except:
                continue

            executable_proposal_names = []

            for proposal_path in (instance_info['proposal_edits_paths'] + [start_file_path, goal_file_path]):
                # Render the images for that proposal_renders_path
                proposal_name = os.path.basename(proposal_path).split('.')[0] # Extract the name of py file, without suffix
                proposal_renders_dir = os.path.join(task_instance_dir, proposal_name)
                
                # Render images. "executable" checks whether the proposal is executable in Blender-Python API.
                if not os.path.exists(proposal_renders_dir) or not os.listdir(proposal_renders_dir): 
                    try:
                        executable = blender_step(infinigen_installation_path, blender_file_path, blender_render_script_path, proposal_path, proposal_renders_dir, merge_all_renders=False, replace_if_overlap=True)
                    except:
                        continue
                    if executable:
                        executable_proposal_names.append((proposal_renders_dir,proposal_name))
                else:
                    executable_proposal_names.append((proposal_renders_dir,proposal_name))
            
            # Loop through each executable proposal to compute their scores
            for proposal_renders_dir, proposal_name in tqdm(executable_proposal_names):
                if proposal_name == 'goal':
                    continue

                task_instance_scores[proposal_name] = {}    

                n_clip_views = []
                pl_views = []    

                for render_name in os.listdir(proposal_renders_dir):
                    task_instance_scores[proposal_name][render_name] = {}

                    # Get path for render
                    proposal_render = Image.open(os.path.join(proposal_renders_dir, render_name))
                    gt_render = Image.open(os.path.join(task_instance_dir, 'goal', render_name))

                    # Compute n_clip and pl
                    n_clip = float(1 - clip_similarity(proposal_render, gt_render))
                    pl = float(photometric_loss(proposal_render, gt_render))

                    # Aggregate scores across all views for a proposal edit to compute average
                    n_clip_views.append(n_clip)
                    pl_views.append(pl)

                    # Record scores for this render
                    task_instance_scores[proposal_name][render_name]['n_clip'] = n_clip
                    task_instance_scores[proposal_name][render_name]['pl'] = pl 
                
                # Compute average n_clip for this proposal
                if n_clip_views:
                    average_n_clip_views = sum(n_clip_views) / len(n_clip_views)

                # Compute average pl for this task instance
                if pl_views:
                    average_pl_views = sum(pl_views) / len(pl_views)

                # Record average scores for a proposal 
                task_instance_scores[proposal_name]['avg_n_clip'] = average_n_clip_views
                task_instance_scores[proposal_name]['avg_pl'] = average_pl_views 
            
                # Save the local scores to the task_instance dir
                task_instance_scores_path = os.path.join(task_instance_dir, 'scores.json')
                with open(task_instance_scores_path, 'w') as file:
                    json.dump(task_instance_scores, file, indent=4)

            # Extract best scores and record them
            best_n_clip_proposal_name = min(task_instance_scores, key=lambda proposal_name: task_instance_scores[proposal_name]['avg_n_clip'])            
            best_pl_proposal_name = min(task_instance_scores, key=lambda proposal_name: task_instance_scores[proposal_name]['avg_pl'])
            best_n_clip = task_instance_scores[best_n_clip_proposal_name]['avg_n_clip']
            best_pl = task_instance_scores[best_pl_proposal_name]['avg_pl']
            task_instance_scores['best_n_clip'] = (best_n_clip_proposal_name, best_n_clip)
            task_instance_scores['best_pl'] = (best_pl_proposal_name, best_pl)
            
            # Register this instance to the scores across this task
            scores_across_instances['best_n_clip'].append(best_n_clip)
            scores_across_instances['best_pl'].append(best_pl)

            # Handle selected edit if applicable
            selected_proposal_path = instance_info['selected_edit_path']
            if selected_proposal_path:
                selected_proposal_name = os.path.basename(selected_proposal_path).split('.')[0]
                selectd_n_clip = task_instance_scores[selected_proposal_name]['avg_n_clip']
                selected_pl = task_instance_scores[selected_proposal_name]['avg_pl']
                task_instance_scores['selected_scores'] =  (selected_proposal_name, {'avg_n_clip':selectd_n_clip, 'avg_pl':selected_pl})
                
                # Register this instance to the scores across this task
                scores_across_instances["selected_n_clip"].append(selectd_n_clip)
                scores_across_instances["selected_pl"].append(selected_pl)

            # Save the local scores to the task_instance dir
            task_instance_scores_path = os.path.join(task_instance_dir, 'scores.json')
            with open(task_instance_scores_path, 'w') as file:
                json.dump(task_instance_scores, file, indent=4)

            scores_across_instances_path = os.path.join(eval_render_save_dir, f'{task}_scores.json',)
            with open(scores_across_instances_path, 'w') as file:
                json.dump(scores_across_instances, file, indent=4)

        # If the model cannot provide any edit for more than 75%
        if len(scores_across_instances['best_n_clip']) < (len(inference_metadata[task]) * 0.25) :
            scores_across_tasks[task] = {}

        # If VLM system doesn't support selection
        elif not scores_across_instances["selected_n_clip"]:
            scores_across_tasks[task] = {
                'best_n_clip': sum(scores_across_instances['best_n_clip']) / len(scores_across_instances['best_n_clip']),
                'best_pl': sum(scores_across_instances['best_pl']) / len(scores_across_instances['best_pl']),
            }

        else: 
            scores_across_tasks[task] = {
                'best_n_clip': sum(scores_across_instances['best_n_clip']) / len(scores_across_instances['best_n_clip']),
                'best_pl': sum(scores_across_instances['best_pl']) / len(scores_across_instances['best_pl']),
                'selected_n_clip': sum(scores_across_instances['selected_n_clip']) / len(scores_across_instances['selected_n_clip']),
                'selected_pl': sum(scores_across_instances['selected_pl']) / len(scores_across_instances['selected_pl']),
            }
        
        intermediates[task] = scores_across_instances
        
    scores_across_tasks_path = os.path.join(eval_render_save_dir, 'overall_scores.json',)
    with open(scores_across_tasks_path, 'w') as file:
        json.dump(scores_across_tasks, file, indent=4)
    
    scores_across_instances_path = os.path.join(eval_render_save_dir, 'intermediate_scores.json',)
    with open(scores_across_instances_path, 'w') as file:
        json.dump(intermediates, file, indent=4)
            

        # Compute Chamfer Distance for 3D-related tasks

        
