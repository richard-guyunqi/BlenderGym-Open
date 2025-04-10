# Our VLM system setup
This is the code for the VLM system set up described in Sec.4.1 and Fig. 2 of the BlenderGym paper. We base our setup on [BlenderAlchemy](https://github.com/ianhuang0630/BlenderAlchemyOfficial). 


# To create your own VLM System
1. Modify function VLMSystem_run() in /inference.py to fit your VLM System in our data flow(input and output format). Instructions are contained in VLMSystem_run().
2. Substitute blender installation path in your code with our infinigen blender. If your VLM system does not involve Blender rendering, ignore this. 
3. Define your own blender render script according to the file storage system of your VLM system. If your VLM system does not involve Blender rendering, ignore this.
4. Run /inference.py to inference your VLM system, results metadata saved to /info_saved.
5. Evaluate your results with /evalaution.py
6. Check scores in /info_saved