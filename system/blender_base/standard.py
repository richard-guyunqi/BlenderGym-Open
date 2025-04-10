"""
Given a python file (indicated inthe commandline path), render the material output.
"""

import bpy
import random
import json
import os
import sys
from sys import platform


# def get_material_from_code(code_fpath):
#     assert os.path.exists(code_fpath)
#     import pdb; pdb.set_trace()
#     with open(code_fpath, "r") as f:
#         code = f.read()
#     exec(code)
#     return material 


if __name__ == "__main__":

    code_fpath = sys.argv[6]  # TODO: allow a folder to be given, each with a possible guess.
    rendering_fpath = sys.argv[7] # rendering

# Enable GPU rendering
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # or 'OPTIX' if your GPU supports it
    bpy.context.preferences.addons['cycles'].preferences.get_devices()

    # Check and select the GPUs
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        if device.type == 'GPU' and not device.use:
            device.use = True

    # Set the rendering device to GPU
    bpy.context.scene.cycles.device = 'GPU'

    # Setting up rendering resolution
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    # Set max samples to 512
    bpy.context.scene.cycles.samples = 512

    # Set color mode to RGB
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    with open(code_fpath, "r") as f:
        code = f.read()
    try:
        exec(code)
    except:
        raise ValueError
    
    # render, and save.
    bpy.context.scene.camera = bpy.data.objects['Camera1']
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = rendering_fpath
    bpy.ops.render.render(write_still=True)
