

import sys
import os
import subprocess
from pathlib import Path
import json
import importlib
from enum import Enum
from bson import ObjectId
from PIL import Image
import random
import shutil
from loguru import logger
import threading
import time
import io

from utils.image import plot_image_grid
from utils.code import get_code_as_string

from tasksolver.event import *
from tasksolver.common import  Question
from tasksolver.exceptions import  CodeExecutionException
from tasksolver.agent import Agent
from tqdm import tqdm

from agents import EditCodeAgent, GeneralAgent, Agent


class TaskSetting(Enum):
    LIGHTING = 1
    MATERIAL = 2
    SHAPEKEY = 3
    GEONODES = 4
    PLACEMENT = 5

TASKSETTING2PROMPTMODULE = {TaskSetting.GEONODES: "geonodes",
                            TaskSetting.LIGHTING: "lighting",
                            TaskSetting.MATERIAL: "material",
                            TaskSetting.SHAPEKEY: "shapekey",
                            TaskSetting.PLACEMENT: "placement"}

def tree_branch(branching_factor:int, question_to_agent:Question, agent:Agent,
                script_save:Path, render_save:Path, thoughtprocess_save:Path,
                blender_file:str, blender_script:str,
                iteration:int, config:dict):
    '''
    For a given question, generate a list of runnable modifications by think and act
    '''

    query_and_act_semaphore = threading.Semaphore(
        min(config["run_config"]["max_concurrent_rendering_processes"], 
            config["run_config"]["max_concurrent_generator_requests"]) )
    
    results = [None] * branching_factor     # each slot is a position for a proposed modification
    def thread(question_to_agent, idx, results):
        # Fill one spot in the list results with a potential code change
        # The code change must be runnable, as checked by agent.act
        with query_and_act_semaphore: 
            #logger.info(f"thread {idx} acquired semaphore lock.")
            done = False
            num_tries = 0 
            max_tries = 1
            while not done and num_tries < max_tries:
                num_tries += 1
                try:
                    # Generate the code by think, the whole trunk
                    p_ans = agent.think(question_to_agent, num_tokens=3000, agent_idx=idx)
                    if len(p_ans.code) == 0:
                        logger.warning(f"The following response didn't parse into any code:\n{idx, script_save}")
                        pass
                except Exception as e: # TODO  ratelimitexception
                    print(e)
                    logger.warning(f"thread {idx} LLM querying failed with error:\n{str(e)}") 
                    time.sleep(30)
                    continue
                try:
                    # Execute the code by act
                    code_path, render_path = agent.act(p_ans, 
                                                    script_save=script_save, 
                                                    render_save=render_save, 
                                                    iteration=iteration,
                                                    blender_file=blender_file,
                                                    blender_script=blender_script,
                                                    config=config,
                                                    blender_step=blender_step)
                    done = True
                except CodeExecutionException:
                    # blender execution failed, count failure.
                    # code_exec_exceptions += 1 
                    #logger.warning(f"thread {idx} code execution failed.")
                    pass
            if not done: 
                code_path = None
                render_path = None 
            results[idx] = (code_path, render_path, 'placeholder')  # The in-place modification of results with the 3-tuple
        logger.info(f"thread {idx} released semaphore lock.")

    # FOR DEBUGGING
    # thread(question_to_agent, 0, results)    

    # Run the think-act pipeline for branching_factor(width) times
    llm_threads = [threading.Thread(target=thread,
                     args=(question_to_agent, i, results))
                     for i in range(branching_factor)]
    for idx, x in enumerate(llm_threads):
        x.start()
        #logger.info(f"starting thread {idx}...")

    for x in llm_threads:
        x.join() # wait till they all finish
    #logger.info(f"joined all threads")
    
    # assert all([el is not None for el in results])
    clean_results = []
    for result in results:
        if result is None:
            clean_results.append((None, None, 'placeholder'))
        else:
            clean_results.append(result)
    
    results = clean_results
    return results


def get_top_candidate(candidates, target, judge, task_setting:TaskSetting, config:dict, 
                            target_description=None, use_vision=True,):

    prompting_submodule = importlib.import_module("prompting."+TASKSETTING2PROMPTMODULE[task_setting])
    craft_eval_question = getattr(prompting_submodule, "craft_eval_question")
    evaluation_semaphore = threading.Semaphore(config["run_config"]["max_concurrent_evaluation_requests"])

    def competition_thread(candidate1, candidate2, results, index, target_image=target):
        # Takes in two candidates and return left or right

        with evaluation_semaphore:
            # randomize the ordering
            done = False
            num_tries = 0
            max_tries = 3

            while not done and num_tries < max_tries:
                num_tries += 1
                order = random.sample([0,1], 2)

                left_code = get_code_as_string([candidate1[0], candidate2[0]][order[0]])
                left_img_file = [candidate1[1], candidate2[1]][order[0]]
                left_img = Image.open(left_img_file)

                right_code = get_code_as_string([candidate1[0], candidate2[0]][order[1]])
                right_img_file = [candidate1[1], candidate2[1]][order[1]]
                right_img = Image.open(right_img_file)

                assert left_img is not None
                assert right_img is not None
                assert left_code is not None
                assert right_code is not None
                if target_description is None:
                    print("target description is None -- intended?")

                question_to_critic = craft_eval_question(
                                        target_image=target_image,
                                        left_image=left_img,
                                        right_image=right_img,
                                        left_code=left_code,
                                        right_code=right_code,
                                        target_description=target_description,
                                        use_vision=use_vision)
                
                if target_description is None and target is None:
                    raise ValueError("No target provided to the competition_thread, either textual or image")

                try: 
                    p_ans = judge.think(question_to_critic, num_tokens=1000, agent_idx=index)
                    done = True
                except Exception as e: # TODO  ratelimitexception
                    logger.warning(f"Sleep for 30s, {str(e)}")
                    time.sleep(30)
                    continue

            if done:
                if p_ans.data == "left":
                    winner_index = order[0]
                elif p_ans.data == "right":
                    winner_index = order[1]
                
                # return a tuple of (winner, (left_img, right_img), raw answer(left or right), question )
                results[index] = ([candidate1, candidate2][winner_index], 
                                (left_img_file, right_img_file),
                                p_ans.raw, 
                                question_to_critic)             
            
    # assert len(candidates)%2 == 0, "Number candidates should be even, otherwise not handled."
    assert len(candidates) > 0, "the candidate list is empty"
     
    odd_one_out = None
    if len(candidates)%2 == 1: # number of candidates is odd
        odd_one_out = candidates[-1] 
        candidates = candidates[:-1]

    num_candidates = len(candidates)
    assert num_candidates % 2 == 0

    results = [None]*(num_candidates//2)

    max_tries = 3
    num_tries = 0
    done = False
    while not done and num_tries < max_tries:
        num_tries += 1 

        if num_tries > 1:
            #logger.warning(f"reattempt # {num_tries} for competition between candidates")
            pass

        # results: each entry is (winner, (left_img, right_img), raw answer(left or right), question) 
        competition_threads = [threading.Thread(target=competition_thread,
                args=(candidates[2*i], candidates[2*i+1], results, i, target))
                for i in range(num_candidates//2)]
        
        for x in competition_threads:
            x.start()
        for x in competition_threads:
            x.join()             

        # Pick the winners from the list results
        winners = [winner[0] for winner in results if winner is not None]
        if len(winners) == 0 and num_candidates//2 > 0:
            continue
        done = True

        intermediates = [{"left": winner[1][0],
                        "right": winner[1][1],
                        "winner": winner[0][1],
                        "inbound_question": str(winner[3]),
                        "thought_string": winner[2]} 
                        for winner in results if winner is not None]

    if not(len(winners) > 0 or num_candidates//2 == 0):
        # raise ValueError("All comparisons between samples seem to have failed.")
        print("All comparisons between samples seem to have failed.")
    
    if odd_one_out is not None:
        winners += [odd_one_out]
    
    try:
        intermediates
    except NameError:
        intermediates = []

    if len(winners) > 1:
        winner, _intermediates = get_top_candidate(winners, target, judge, config=config,
                    target_description=target_description, task_setting=task_setting,
                    use_vision=use_vision) 
        return winner, intermediates + _intermediates
    else:
        if not winners:
            winners.append((None, None))
        return winners[0], intermediates # the only winner


def make_if_nonexistent(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

# def blender_step(config, blender_file, blender_script, script_path, render_path, 
#                 verify_render_path=True):

#     '''
#     Generate a rendered image with given script_path at render_path
#     '''

#     if verify_render_path  and os.path.isfile(render_path):
#         raise ValueError(f"verify_render_path is True but {render_path} already exists before blender process.")

#     assert blender_file is not None and blender_script is not None
    
#     # Enter the blender code
#     command = [config["run_config"]["blender_command"], "--background", blender_file, 
#                     "--python", blender_script, 
#                     "--", script_path, render_path]
#     command = ' '.join(command)
#     command_run = subprocess.run(command, shell=True, check=True)
    
#     if verify_render_path  and not os.path.isfile(render_path):
#         logger.warning(f"The following bpy script didn't run correctly in blender:{script_path}")
#         raise CodeExecutionException 

#     return None


def merge_images_in_directory(directory, saved_to_local=True, merge_dir_into_image=True):
    '''
    Merge all images in the given directory into a single image.
    '''
    # Get a list of image paths
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg', 'webp'))]

    if not image_paths:
        new_image = None
        images=[]
        shutil.rmtree(directory)
        raise CodeExecutionException
    else:
        # Open images and get their sizes
        images = [Image.open(img) for img in image_paths]
        widths, heights = zip(*(i.size for i in images))

        # Calculate total size for the final image
        total_width = sum(widths)
        max_height = max(heights)

        # Create a new blank image with the calculated size
        new_image = Image.new('RGB', (total_width, max_height))

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

    # if is_directory_empty(render_dir):
    #     print(f"The following bpy script didn't run correctly in blender:{script_path}")
    #     return False
    #     # raise CodeExecutionException 
    # else:
    #     if merge_all_renders:
    merge_images_in_directory(render_dir, saved_to_local=True, merge_dir_into_image=merge_dir_into_image)
    return os.path.exists(render_dir)

    return True


def refinement(config, credentials, breadth, depth, blender_file, blender_script, 
                init_code, method_variation, output_folder, overwrite=True):        
    
    run_config = config["run_config"]

    # define task type and import relevant prompting modules 
    task_type = getattr(TaskSetting, config["task"]["type"].upper())
    prompting_submodule = importlib.import_module("prompting."+TASKSETTING2PROMPTMODULE[task_type])

    
    # import task definitions 
    code_editing_task = getattr(prompting_submodule, "code_editing_task")
    parameter_search_task = getattr(prompting_submodule, "parameter_search_task")
    pruning_task = getattr(prompting_submodule, "pruning_task")
    craft_tuner_question = getattr(prompting_submodule, "craft_tuner_question")
    craft_leap_question =  getattr(prompting_submodule, "craft_leap_question")


    # Read in one line of target description(instruction)
    target_description = None
    if config["input"]["text_prompt"] is not None:
        target_description = config["input"]["text_prompt"]

    assert method_variation in ("tune_leap", "leap", "tune")

    # Create all the dirs for results
    script_save = Path(output_folder)/Path("scripts/")
    render_save = Path(output_folder)/Path("renders/")
    thoughtprocess_save = Path(output_folder)/Path("thought_process/")
    make_if_nonexistent(output_folder)
    make_if_nonexistent(script_save)
    make_if_nonexistent(render_save)
    make_if_nonexistent(thoughtprocess_save)
            
    init_render_file = os.path.join(output_folder, "init_render.png")      # The original blender rendered image
    target_code = config["input"]["target_code"]       # Target bpy code
    target_render_file = config["input"]["input_image"]     # Dalle generated pseudo-target image based on the text file
    
    assert init_code is not None
    if not os.path.exists(init_render_file):
        # blender_step(config, blender_file, blender_script, init_code, init_render_file, verify_render_path=True)        
        blender_step(config["run_config"]["blender_command"], blender_file, blender_script, init_code, init_render_file, merge_all_renders=True, merge_dir_into_image=True)

    init_image = Image.open(init_render_file)      # Keep an record of original image


    if target_render_file is not None:      # If provided with a path to ideal target image
        if target_code is not None and not os.path.exists(target_render_file):  # If target_code is also provided and no image provided
            # blender_step(config, blender_file, blender_script, target_code, target_render_file,  verify_render_path=True)  # Render and overwrite the dalle generated images
            blender_step(config["run_config"]["blender_command"], blender_file, blender_script, target_code, target_render_file, merge_all_renders=True, merge_dir_into_image=True)

        target_image = Image.open(target_render_file)      
    else:
        target_image = None

    # create agents 
    thinker_is_visual = None
    evaluator_is_visual = None
    
    # Initialize the thinker (code editor and parameter searcher)
    thinker_class_type = GeneralAgent if config["run_config"]["edit_style"] == "rewrite_code" else EditCodeAgent 

    # if run_config["edit_generator_type"] in ("gpt-4o", "o1", "o1-mini", "gpt-4-turbo", 'gpt-4o-mini', 'o3-mini', 'gpt-4-turbo'): 
    #     # Default!
    #     param_tuner = thinker_class_type (credentials["openai"], parameter_search_task, vision_model=run_config["edit_generator_type"])
    #     agent = thinker_class_type (credentials["openai"], code_editing_task, vision_model=run_config["edit_generator_type"])
    #     thinker_is_visual = True

    # elif run_config["edit_generator_type"] in ("claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"):
    #     # tasks stay the same.
    #     param_tuner = thinker_class_type (credentials["claude"], parameter_search_task, vision_model=run_config["edit_generator_type"])
    #     agent = thinker_class_type (credentials["claude"], code_editing_task, run_config["edit_generator_type"])
    #     thinker_is_visual = True

    # elif run_config["edit_generator_type"] in ('gemini-pro', 'gemini-pro-vision', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'):
    #     # tasks stay the same.
    #     param_tuner = thinker_class_type (credentials["gemini"], parameter_search_task, vision_model=run_config["edit_generator_type"])
    #     agent = thinker_class_type (credentials["gemini"], code_editing_task, run_config["edit_generator_type"])
    #     thinker_is_visual = True

    # elif run_config["edit_generator_type"] == "Intern":
    #     # tasks stay the same.
    #     param_tuner = thinker_class_type (None, parameter_search_task, vision_model='OpenGVLab/InternVL2-8B')
    #     agent = param_tuner
    #     thinker_is_visual = True

    # elif run_config["edit_generator_type"] == "InternLlama":
    #     # tasks stay the same.
    #     param_tuner = thinker_class_type (None, parameter_search_task, vision_model=run_config["edit_generator_type"])
    #     agent = param_tuner
    #     thinker_is_visual = True

    # elif run_config["edit_generator_type"] == "Qwen":
    #     # tasks stay the same.
    #     param_tuner = thinker_class_type (None, parameter_search_task, vision_model='Qwen/Qwen2-VL-7B-Instruct-AWQ')
    #     agent = param_tuner
    #     thinker_is_visual = True

    # elif run_config["edit_generator_type"] == "QwenLlama":
    #     # tasks stay the same.
    #     param_tuner = thinker_class_type (None, parameter_search_task, vision_model=run_config["edit_generator_type"])
    #     agent = param_tuner
    #     thinker_is_visual = True

    # elif run_config["edit_generator_type"] == "MiniCPM":
    #     # tasks stay the same.
    #     param_tuner = thinker_class_type (None, parameter_search_task, vision_model='openbmb/MiniCPM-V-2_6-int4')
    #     agent = param_tuner
    #     thinker_is_visual = True

    # elif run_config["edit_generator_type"] == "MiniCPMLlama":
    #     # tasks stay the same.
    #     param_tuner = thinker_class_type (None, parameter_search_task, vision_model=run_config["edit_generator_type"])
    #     agent = param_tuner
    #     thinker_is_visual = True

    # elif run_config["edit_generator_type"] == "Phi":
    #     # tasks stay the same.
    #     param_tuner = thinker_class_type (None, parameter_search_task, vision_model='microsoft/Phi-3.5-vision-instruct')
    #     agent = param_tuner
    #     thinker_is_visual = True

    # elif run_config["edit_generator_type"] == "PhiLlama":
    #     # tasks stay the same.
    #     param_tuner = thinker_class_type (None, parameter_search_task, vision_model=run_config["edit_generator_type"])
    #     agent = param_tuner
    #     thinker_is_visual = True
    # else:
    #     param_tuner = thinker_class_type (None, parameter_search_task, vision_model=run_config["edit_generator_type"])

    #     # TODO: If your model requires API, you can add credentials["your_model"]
    #     # param_tuner = thinker_class_type (credentials["your_model"], parameter_search_task, vision_model=run_config["edit_generator_type"])
    #     agent = param_tuner
    #     thinker_is_visual = True

    param_tuner = thinker_class_type (credentials, parameter_search_task, vision_model=run_config["edit_generator_type"])
    agent = thinker_class_type (credentials, code_editing_task, vision_model=run_config["edit_generator_type"])
    thinker_is_visual = True


    # # Initialize the state evaluator (pruning task)
    # if run_config["state_evaluator_type"] in ("gpt-4o", "o1", "o1-mini", "gpt-4-turbo", "gpt-4o-mini", 'o3-mini', 'gpt-4-turbo'): 
    #     judge = GeneralAgent(credentials, pruning_task, vision_model=run_config["state_evaluator_type"])
    #     evaluator_is_visual = True

    # elif run_config["state_evaluator_type"] in ("claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest", "claude-3-5-sonnet-20240620"):
    #     judge = GeneralAgent(credentials, pruning_task, vision_model=run_config["state_evaluator_type"])
    #     evaluator_is_visual = True

    # elif run_config["state_evaluator_type"] in ('gemini-pro', 'gemini-pro-vision', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'):
    #     judge = GeneralAgent(credentials, pruning_task, vision_model=run_config["state_evaluator_type"])
    #     evaluator_is_visual = True

    # elif run_config["state_evaluator_type"] in ("Intern", "InternLlama"):
    #     judge = GeneralAgent(credentials, pruning_task, vision_model=run_config["state_evaluator_type"])
    #     evaluator_is_visual = True

    # elif run_config["state_evaluator_type"] in ("MiniCPM", "MiniCPMLlama"):
    #     judge = GeneralAgent(None, pruning_task, vision_model='openbmb/MiniCPM-V-2_6-int4')
    #     evaluator_is_visual = True

    # elif run_config["state_evaluator_type"] in ("Qwen", "QwenLlama"):
    #     judge = GeneralAgent(None, pruning_task, vision_model='Qwen/Qwen2-VL-7B-Instruct-AWQ')
    #     evaluator_is_visual = True

    # elif run_config["state_evaluator_type"] in ("Phi", "PhiLlama"):
    #     judge = GeneralAgent(None, pruning_task, vision_model='microsoft/Phi-3.5-vision-instruct')
    #     evaluator_is_visual = True

    # else:
    #     judge = GeneralAgent(None, pruning_task, vision_model=run_config["state_evaluator_type"])
    #     evaluator_is_visual = True

    judge = GeneralAgent(credentials, pruning_task, vision_model=run_config["state_evaluator_type"])
    evaluator_is_visual = True

    # raise ValueError(f"Invalid evaluator: {run_config['state_evaluator_type']}")

    assert thinker_is_visual is not None and evaluator_is_visual is not None, "Remember to assign these variables above."
    print(evaluator_is_visual, thinker_is_visual)
    if not evaluator_is_visual or not thinker_is_visual:
        assert target_description is not None, f"When either the {run_config['edit_generator_type']} or {run_config['state_evaluator_type']} is not visual, make sure that you have the target description."

    

    # start of simulation
    code_path = init_code       # original starter code
    render_path = init_render_file      # path of original rendered image

    intermediary_outputs = []

    for i in tqdm(range(depth)):       # Tree depth
        if not overwrite:
            if os.path.exists(thoughtprocess_save/f"iteration_{i}.json"):
                with open(thoughtprocess_save/f"iteration_{i}.json", "r") as f: 
                    process_json = json.load(f)

                # If there is a winner among the last layer of branches,
                # keep them since we will compare that to the output of this layer
                if "winner_code" in process_json[-1] and "winner_image" in process_json[-1]:
                    code_path = process_json[-1]["winner_code"]
                    render_path = process_json[-1]["winner_image"]
                    intermediary_outputs.append({
                                    'code_path': code_path, 
                                    'render_path': render_path,
                                    "iteration": i})
                    logger.info(f"Thought process loaded for iteration {i}")
                    continue 
                else:
                    pass # start at this iteration! 

        process_json = []
        
        if (method_variation in ('tune_leap',) and i%2 == 0) or method_variation in ('tune',):
        # The case of tune

            # Craft a question based on the image and text input
            tuner_question = craft_tuner_question(
                blender_init_code_str=get_code_as_string(code_path),
                init_image=Image.open(render_path), 
                target_image=target_image,
                target_description=target_description,
                use_vision=thinker_is_visual)
            
            logger.info(f"tuner_question_formed")

            # Run think-act on the agent by `breadth` times
            # results is a list of length `breadth`, one runnable modification on each entry
            # Each entry is (code_path, render_path, p_ans.raw), code_path is the path to the modified bpy script, 
            # render_path the resulting rendered image, and parsed answer
            results = tree_branch(breadth, tuner_question, 
                                        agent=param_tuner,
                                        script_save=script_save,
                                        render_save=render_save,
                                        thoughtprocess_save=process_json,
                                        blender_file=blender_file, 
                                        blender_script=blender_script,
                                        iteration=i, config=config)
            
            logger.info(f"Runnable modifications generated for iteration {i}/{depth-1}(0-indexed) of depth")

            results = [el for el in results if el[0] is not None]       # Take out the code_path

            # Register all the potential modifications to the json file
            process_json.append(
                {
                    "phase": "explode_options_TUNE",
                    "iteration": i,
                    "inbound_question": str(tuner_question),
                    "choices_image": [res[1] for res in results],
                    "choices_code": [res[0] for res in results],
                    "thought_strings": [res[2] for res in results]
                }   
            )

            results.append((code_path, render_path, 'placeholder'))

            # Get the top candidate by state evaluator
            if len(results) > 1:

                # top_candidate is a (code, image) pair
                top_candidate, intermediates = get_top_candidate(results,
                                target_image, judge, config=config, 
                                target_description=target_description, 
                                task_setting=task_type,
                                use_vision=evaluator_is_visual)
                process_json.append(
                    {
                        "phase": "selection",
                        "choices_image": [res[1] for res in results],
                        "choices_code": [res[0] for res in results],
                        "winner_image": top_candidate[1],
                        "winner_code": top_candidate[0],
                        "decision_process": intermediates
                    }
                )
            else:
                top_candidate = (code_path, render_path)
                process_json.append(
                    {
                        "phase": "selection",
                        "choices_image": [res[1] for res in results],
                        "choices_code": [res[0] for res in results],
                        "winner_image": top_candidate[1],
                        "winner_code": top_candidate[0],
                        "decision_process": None
                    }
                )
            
        elif (method_variation in ('tune_leap',) and i%2 == 1) or method_variation in ('leap',):
            # Similar to tune mode, but takes bigger step for the agent to think

            # we leap
            #  Craft a question based on the image and text input
            question_to_agent = craft_leap_question(
                blender_init_code_str = get_code_as_string(code_path),
                init_image = Image.open(render_path),
                target_image=target_image,
                target_description=target_description,
                use_vision=thinker_is_visual) 

            results = tree_branch(breadth, question_to_agent, 
                                        agent=agent,
                                        script_save=script_save,
                                        render_save=render_save,
                                        thoughtprocess_save=process_json,
                                        blender_file=blender_file, 
                                        blender_script=blender_script,
                                        iteration=i, config=config)
            logger.info(f"Runnable modifications generated for iteration {i}/{depth} of depth")

            results = [el for el in results if el[0] is not None]
            process_json.append(
                {
                    "phase": "explode_options_LEAP",
                    "iteration": i,
                    "inbound_question": str(question_to_agent),
                    "choices_image": [res[1] for res in results],
                    "choices_code": [res[0] for res in results],
                    "thought_strings": [res[2] for res in results]
                }   
            )

            results.append((code_path, render_path, 'placeholder'))

            if len(results) > 1:
                top_candidate, intermediates = get_top_candidate(results, 
                                target_image, judge, config=config, 
                                target_description=target_description,
                                task_setting=task_type,
                                use_vision=evaluator_is_visual)
                process_json.append(
                    {
                        "phase": "selection",
                        "choices_image": [res[1] for res in results],
                        "choices_code": [res[0] for res in results],
                        "winner_image": top_candidate[1],
                        "winner_code": top_candidate[0],
                        "iteration": i,
                        "decision_process": intermediates
                    }
                )
            else:
                top_candidate = (code_path, render_path)
                process_json.append(
                    {
                        "phase": "selection",
                        "choices_image": [res[1] for res in results],
                        "choices_code": [res[0] for res in results],
                        "winner_image": top_candidate[1],
                        "winner_code": top_candidate[0],
                        "decision_process": None
                    }
                )
                
        # if config["run_config"]["enable_hypothesis_reversion"]:          # Compares the current one with the best of last layer
        #     if len(results) > 0:
        #         top_candidate_before = top_candidate
        #         top_candidate, intermediates = get_top_candidate([top_candidate, (code_path, render_path)], 
        #                         target_image, judge, config=config, target_description=target_description,
        #                         task_setting=task_type,
        #                         use_vision=evaluator_is_visual)
        #         process_json.append(
        #             {
        #                 "phase": "selection",
        #                 "choices_image": [top_candidate_before[1], render_path],
        #                 "choices_code": [top_candidate_before[0], code_path],
        #                 "winner_image": top_candidate[1],
        #                 "winner_code": top_candidate[0],
        #                 "iteration": i,
        #                 "decision_process": intermediates
        #             }
        #         ) 

        #         # If last layer's output is better, revert.
        #         if top_candidate != top_candidate_before:
        #             #logger.info("Check shows that old sample is better. Rebasing to old sample.")
        #             pass
        
        logger.info(f"Top candidated picked for iteration {i}/{depth-1}(0-indexed) of depth. Code:{top_candidate[0]}, image:{top_candidate[1]}")

        with open(thoughtprocess_save/f"iteration_{i}.json", "w") as f:
            json.dump(process_json, f)
            logger.info(f"Thought process saved for iteration {i}")

        # set new, best so far
        code_path, render_path = top_candidate[0], top_candidate[1] 
        # add to final output
        intermediary_outputs.append({'code_path': code_path, 
                                'render_path': render_path,
                                "iteration": i})


    fig = plot_image_grid([(Image.open(el["render_path"]) if el is not None else None) 
                for el in intermediary_outputs], 
                rows=1, cols=len(intermediary_outputs))
    fig.savefig(str(output_folder/"best_of.png"))
    
