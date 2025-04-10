import os
import sys
import subprocess
import argparse
import time
import json
from PIL import Image
import shutil
from torchvision import transforms


env = os.environ.copy()

## Focus on model swapping; make a default_BA.py (all BA-based structure) that can reproduce our results, also allow customzied system 
## 

def BlenderAlchemy_run(blender_file_path, start_script, start_render, goal_render, blender_render_script_path, task_instance_id, task, infinigen_installation_path, generator_type, evaluator_type, starter_time=None, tree_dims=(4, 8)):
    '''
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
        proposal_renders_paths: a dictionary with proposal_edits_paths as keys and paths to their respective renders as values 
        selected_edit_path[optional]: if applicable, the file path to the VLM-system-selected proposal script
        selected_render_path[optional]: if applicable, the dir path to the renders of the VLM-system-selected proposal script
    '''

    task_translate = {
            'geometry': 'geonodes',
            'material': 'material',
            'blendshape': 'shapekey',
            'placement': 'placement',
            'lighting': 'lighting'
    }

    task = task_translate[task]
    variants = ['tune_leap']

    # To automatically differentiate the inference results
    if starter_time:
        output_folder_name = f"outputs/outputs_{starter_time}"
    else:
        output_folder_name = "outputs/outputs_test"

    config_dict = {     # This should allow plug-in for different models
        'task':{'type': task},
        'credentials':{
            'openai': 'credentials/openai_api.txt',
            'claude': 'credentials/claude_api.txt',
            'gemini': 'credentials/gemini_api.txt',
        },
        'input':{
            'text_prompt': None,
            'input_image': f'{goal_render}/render1.png',
            'target_code': None,
        },
        'output':{
            'output_dir': f"{output_folder_name}/{task_instance_id}/"
        },
        'run_config':{
            'blender_command': infinigen_installation_path,
            'edit_style': "edit_code",
            'num_tries': 1,
            'enable_visual_imagination': False, 
            'enable_hypothesis_reversion': True,
            'variants': variants,
            'tree_dims': [
                f"{tree_dims[0]}x{tree_dims[1]}"
            ],
            'edit_generator_type': generator_type,
            'state_evaluator_type': evaluator_type,
            'max_concurrent_rendering_processes': 1,
            'max_concurrent_evaluation_requests': 1,
            'max_concurrent_generator_requests': 1
        }
    }
    import yaml
    config_file_path = os.path.abspath('temp.yml')

    with open(config_file_path, 'w') as file:
        yaml.dump(config_dict, file)

    command = f'''
        cd system && \

        python main.py \
            --starter_blend {blender_file_path} \
            --blender_base {blender_render_script_path} \
            --blender_script {start_script} \
            --config {config_file_path}
    '''

    print(f'config_dict: {config_dict}')
    print(f'command: {command}')

    subprocess.run(command, shell=True, env=env)

    proposal_edits_dir_path = f'system/{output_folder_name}/{task_instance_id}/instance0/{variants[0]}_d{tree_dims[0]}_b{tree_dims[1]}/scripts'
    proposal_renders_dir_path = f'system/{output_folder_name}/{task_instance_id}/instance0/{variants[0]}_d{tree_dims[0]}_b{tree_dims[1]}/renders'
    proposal_edits_paths = [os.path.join(proposal_edits_dir_path, edit_path) for edit_path in os.listdir(proposal_edits_dir_path)]
    proposal_renders_paths = [os.path.join(proposal_renders_dir_path, render_path) for render_path in os.listdir(proposal_renders_dir_path)]

    # TEST: Selectd edit for each iteration
    last_iter_info = f'system/{output_folder_name}/{task_instance_id}/instance0/{variants[0]}_d{tree_dims[0]}_b{tree_dims[1]}/thought_process/iteration_{tree_dims[0]-1}.json'
    with open(last_iter_info, 'r') as file:
        info = json.load(file)
    
    selected_edit_path = "system/" + info[-1]['winner_code']
    selected_render_path = "system/" + info[-1]['winner_image']

    return proposal_edits_paths, proposal_renders_paths, selected_edit_path, selected_render_path

def merge_images_in_directory(directory, saved_to_local=True, merge_dir_into_image=True):
    '''
    Merge all images in the given directory into a single image.
    '''
    # Get a list of image paths
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg', 'webp'))]

    # Open images and get their sizes
    images = [Image.open(img) for img in image_paths]
    widths, heights = zip(*(i.size for i in images))

    # Calculate total size for the final image
    total_width = sum(widths)
    max_height = max(heights)

    # Create a new blank image with the calculated size
    if total_width != 0 and max_height != 0:
        new_image = Image.new('RGB', (total_width, max_height))
    else:
        new_image = None

    # Paste all images into the new image
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    if saved_to_local:
        # Save the final image to local
        if not merge_dir_into_image:    # Preserve the dir, adding new image to the dir
            merged_image_path = os.path.join(directory, 'merged_image.png')
        else:   # Delete the dir, and save the merged image as the name of the dir
            shutil.rmtree(directory)
            merged_image_path = directory

        if new_image:
            new_image.save(merged_image_path)
            print(f"Merged image saved to {merged_image_path}")
        return new_image, merged_image_path
    else:
        return new_image, None



def blender_step(infinigen_installation_path, blender_file_path, blender_render_script_path, script_path, render_dir, merge_all_renders=False, replace_if_overlap=True, merge_dir_into_image=False):

    '''
    Generate a rendered image with given script_path at render_dir.

    Inputs:
        blender_file_path: file path to the .blend base file
        blender_render_script_path: file path to the render script of blender scene
        script_path: file path to the script we want to render
        render_dir: dir path to save the rendered images
        merge_all_renders[optional]: True will merge all images in render_dir
        replace_if_overlap[optional]: False will skip if the render_dir exists and is non-empty, and True will proceed replace every overlapping render 
        merge_dir_into_image[optional]: True will delete the render_dir and replace it with the merged image
    '''

    def is_directory_empty(directory_path):
        # Check if the directory exists and is indeed a directory
        if not os.path.isdir(directory_path):
            raise ValueError(f"{directory_path} is not a valid directory path.")
    
        # List the contents of the directory
        return len(os.listdir(directory_path)) == 0

    assert blender_file_path is not None and blender_render_script_path is not None

    if replace_if_overlap:  # Just overwrite the files
        os.makedirs(render_dir, exist_ok=True)
    else:   
        if os.path.isdir(render_dir) and not is_directory_empty(render_dir): # If such dir already exists and is non-empty, skip
            return None 

        os.makedirs(render_dir, exist_ok=True)

    print('blender_render_script_path: ', blender_render_script_path)
    print('script_path: ', script_path)
    print('render_dir: ', render_dir)

    # Enter the blender code
    command = [infinigen_installation_path, "--background", blender_file_path, 
                    "--python", blender_render_script_path, 
                    "--", script_path, render_dir]
    command = ' '.join(command)
    command_run = subprocess.run(command, shell=True, check=True)

    if is_directory_empty(render_dir):
        print(f"The following bpy script didn't run correctly in blender:{script_path}")
        return False
        # raise CodeExecutionException 
    else:
        if merge_all_renders:
            merge_images_in_directory(render_dir, saved_to_local=True, merge_dir_into_image=merge_dir_into_image)

    return True

import sys
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel



def clip_similarity(image1, image2):
    """
    Compute the CLIP similarity between two PIL images.

    Args:
    image1 (PIL.Image): The first input image.
    image2 (PIL.Image): The second input image.

    Returns:
    float: The CLIP similarity between the two images.
    """
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)

    # Load the CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Load the CLIP processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Preprocess the images
    images = [image1, image2]
    inputs = processor(images=images, return_tensors="pt")

    # Compute the features for the images
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    # Compute the cosine similarity between the image features
    sim = torch.nn.functional.cosine_similarity(features[0], features[1], dim=-1)

    return sim.item()

def photometric_loss(image1:Image.Image, image2:Image.Image) -> float:
    """
    Compute the photometric loss between two PIL images.

    Args:
    image1 (PIL.Image): The first input image.
    image2 (PIL.Image): The second input image.

    Returns:
    float: The photometric loss between the two images.
    """

    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    
    # Convert images to numpy arrays
    img1_array = np.array(image1)[:, :, :3]
    img2_array = np.array(image2)[:, :, :3]

    # Normalize images to [0, 1]
    img1_norm = img1_array.astype(np.float32) / 255.0
    img2_norm = img2_array.astype(np.float32) / 255.0

    # Compute the squared difference between the normalized images
    diff = np.square(img1_norm - img2_norm)

    # Compute the mean squared error
    mse = np.mean(diff)
    return mse

    
def img2text_clip_similarity(image, text):
    """
    Compute the CLIP similarity between a PIL image and a text.

    Args:
    image (PIL.Image): The input image.
    text (str): The input text.

    Returns:
    float: The CLIP similarity between the image and the text.
    """
    
    # Load the CLIP model
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    # Load the CLIP processor
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Preprocess the image and text
    inputs = processor(text=text, images=image, return_tensors="pt")

    # Compute the features for the image and text
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs.pixel_values)
        text_features = model.get_text_features(input_ids=inputs.input_ids)

    # Compute the cosine similarity between the image and text features
    sim = torch.nn.functional.cosine_similarity(image_features, text_features, dim=-1)

    return sim.item()

    
def img2img_clip_similarity(image1, image2):
    """
    Compute the CLIP similarity between two PIL images.

    Args:
    image1 (PIL.Image): The first input image.
    image2 (PIL.Image): The second input image.

    Returns:
    float: The CLIP similarity between the two images.
    """
    
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)

    # Load the CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Load the CLIP processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # # Preprocess the images
    images = [image1, image2]
    # images = torch.tensor(images, dtype=torch.float32)  # Explicit dtype
    # inputs = processor(images=images, return_tensors="pt")

    # Define a transform that converts PIL images to tensors
    transform = transforms.ToTensor()

    # Convert images
    images = [transform(image) for image in images]  # Converts each PIL image to a tensor

    # Stack into a batch (Assuming both images have the same size)
    images = torch.stack(images)  

    inputs = processor(images=images, return_tensors="pt")

    # Compute the features for the images
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    # Compute the cosine similarity between the image features
    sim = torch.nn.functional.cosine_similarity(features[0], features[1], dim=-1)

    return sim.item()

def tree_dim_parse(tree_dims):
    try:
        depth, breadth = tree_dims.split('x')
        return (int(depth), int(breadth))
    except:
        raise ValueError('The tree_dims input format is not correct! Please make sure you enter something like "dxb". ')

# class VLMSystem():
#     def __init__(self) -> None:
#         self.env = os.environ.copy()

#     def run(self, blender_file_path, start_script, start_render, goal_render, blender_render_script_path, task_instance_id, task, infinigen_installation_path):
#         '''
#         Generation and potentially selection process of the VLM system.

#         Inputs:
#             blender_file_path: file path to the .blend base file
#             start_file_path: file path to the start.py, the script for start scene
#             start_render_path: dir path to the rendered images of start scene
#             goal_render: dir path to the rendered images of goal scene
#             blender_render_script_path: file path to the render script of blender scene
#             task: name of the task, like `geometry`, `placement`
#             task_instance_id: f'{task}{i}', like `placement1`, `geometry2`
#             infinigen_installation_path: file/dir path to infinigen blender executable file for background rendering

#         Outputs:
#             proposal_edits_paths: a list of file paths to proposal scripts from the VLM system 
#             proposal_renders_paths: a dictionary with proposal_edits_paths as keys and paths to their respective renders as values 
#             selected_edit_path[optional]: if applicable, the file path to the VLM-system-selected proposal script
#             selected_render_path[optional]: if applicable, the dir path to the renders of the VLM-system-selected proposal script
#         '''

#         task_translate = {
#                 'geometry': 'geonodes',
#                 'material': 'material',
#                 'blendshape': 'shapekey',
#                 'placement': 'placement',
#                 'lighting': 'lighting'
#         }

#         task = task_translate[task]

#         config_dict = {
#             'task':{'type': task},
#             'credentials':{
#                 'openai': '/home/richard/Documents/system/openai_api.txt'
#             },
#             'input':{
#                 'text_prompt': None,
#                 'input_image': f'{start_render}/render1.png',
#                 'target_code': None,
#             },
#             'output':{
#                 'output_dir': f"output/{task_instance_id}/"
#             },
#             'run_config':{
#                 'blender_command': infinigen_installation_path,
#                 'edit_style': "rewrite_code",
#                 'num_tries': 1,
#                 'enable_visual_imagination': False, 
#                 'enable_hypothesis_reversion': True,
#                 'variants': [
#                     "tune"
#                 ],
#                 'tree_dims': [
#                     "2x2"
#                 ],
#                 'edit_generator_type': "GPT4V",
#                 'state_evaluator_type': "GPT4V",
#                 'max_concurrent_rendering_processes': 4,
#                 'max_concurrent_evaluation_requests': 2,
#                 'max_concurrent_generator_requests': 4
#             }
#         }
#         import yaml
#         config_file_path = '/home/richard/Documents/blendergym_test/temp.yml'

#         with open(config_file_path, 'w') as file:
#             yaml.dump(config_dict, file)

#         command = f'''
#             cd /home/richard/Documents/system && \

#             python main.py \
#                 --starter_blend {blender_file_path} \
#                 --blender_base {blender_render_script_path} \
#                 --blender_script {start_script} \
#                 --config {config_file_path}
#         '''

#         # print(f'config_dict: {config_dict}')
#         # print(f'command: {command}')

#         # subprocess.run(command, shell=True, env=self.env)

#         proposal_edits_dir_path = f'/home/richard/Documents/system/output/{task_instance_id}/instance0/tune_d2_b2/scripts'
#         proposal_renders_dir_path = f'/home/richard/Documents/system/output/{task_instance_id}/instance0/tune_d2_b2/renders'
#         proposal_edits_paths = [os.path.join(proposal_edits_dir_path, edit_path) for edit_path in os.listdir(proposal_edits_dir_path)]
#         proposal_renders_paths = [os.path.join(proposal_renders_dir_path, render_path) for render_path in os.listdir(proposal_renders_dir_path)]
#         # selected_edit_path = '/home/richard/Documents/system/output/task_instance_id/instance0/tune_d2_b3/renders'
#         # selected_render_path = '/home/richard/Documents/system/output/task_instance_id/instance0/tune_d2_b3/renders'
#         return proposal_edits_paths, proposal_renders_paths, None, None

