# ViewPlanningToolbox
A lightweight View Planning Toolbox (VPT) for simulated image acquisition, trajectory plans, trajectory visualization, and 3D reconstruction


Pre-requisites
--------------
1. Ubuntu 20.04 or compatible version (Tested with 20.04)
2. Python 3.8.0 or compatible version (Tested with 3.8.0)
3. Blender 3.5.1 or compatible version (Tested with 3.5.1)
4. VirtualEnv for the package management
Installation Instructions
-------------------------
1. Clone this repository into your python project folder using terminal:
```
git clone https://github.com/matthew-tucsok/ViewPlanningToolbox.git <your_project_folder>
```
2. Install the requirements.txt
```
cd ViewPlanningToolbox
pip install -r requirements.txt
```
3. Add Blender as an alias to your .bashrc file:
```
nano ~/.bashrc
```
Add the following line to the end of the file:
```
alias blender='/<your_blender_path>/blender'
```

4. Follow the installation instructions  steps 3-5 for [BlendTorch](https://github.com/cheind/pytorch-blender) replacing <DST> with "pytorch-blender".  

5. Modify config.ini to point to your blender executable and change your camera settings
