# ViewPlanningToolbox
A lightweight View Planning Toolbox (VPT) for simulated image acquisition, trajectory plans, trajectory visualization, and 3D reconstruction

Citation
--------------
If you use the code or the toolbox for your work, please cite the following paper:
```
@article{gazani2023bag,
  title={Bag of views: An appearance-based approach to next-best-view planning for 3d reconstruction},
  author={Gazani, Sara Hatami and Tucsok, Matthew and Mantegh, Iraj and Najjaran, Homayoun},
  journal={IEEE Robotics and Automation Letters},
  volume={9},
  number={1},
  pages={295--302},
  year={2023},
  publisher={IEEE}
}
```

Pre-requisites
--------------
1. Ubuntu 20.04 or compatible version (Tested with 20.04)
2. Python 3.8.0 or compatible version (Tested with 3.8.0)
3. Blender 3.5.1 or compatible version (Tested with 3.5.1)
4. VirtualEnv for the package management
Installation Instructions
-------------------------
1. Clone this repository into your Python project folder using the terminal:
```
git clone https://github.com/matthew-tucsok/ViewPlanningToolbox.git <your_project_folder>
```
2. Install the requirements.txt.
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

5. Modify config.ini to point to your blender executable and change your camera settings.
