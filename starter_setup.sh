# Install basic pacakges
pip install torch
cd TaskSolver
pip install -e .
cd ../

# Download benchdata.zip and copy that under /BlenderGym
python generate_benchdata.py

# Install Infinigen package
git clone git@github.com:richard-guyunqi/infinigen.git
cd infinigen
INFINIGEN_MINIMAL_INSTALL=True bash scripts/install/interactive_blender.sh
cd ..

