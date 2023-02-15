# PyDScat-GPU
This package is used for the rank-one decomposition to accelerate the solution of the discrete-dipole approximation (RD-DDA) with GPU. A concrete example for the mathematical formulation is [here](./Mathematica/DDA_RankOne_Decomp.htm). See [here](./pydscat/README.md) for the definitions of the parameters.
## Environment requirement
* Python >= 3.6 ([Linux](http://docs.python-guide.org/en/latest/starting/install3/linux/)) (Only Linux environment is supported due to the plot tool)
## Python library
* [fresnel](https://fresnel.readthedocs.io/en/stable/installation.html): please install this package with conda instead of pip.
```python
conda install -c conda-forge fresnel
```
* [Pillow](https://pillow.readthedocs.io/en/stable/)
```python
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade Pillow
```
* [Tensorflow >= 2.0](https://www.tensorflow.org/) (we used Tensorflow 2.1.0 mainly, where the efficiency was benchmarked in this version)
```python
# Requires the latest pip
pip install --upgrade pip
# Current stable release for CPU and GPU
pip install tensorflow
```
* [NetworkX](https://networkx.org/)
```python
pip install networkx
```

## Example code
### [Example 1](./Examples/example1): simulation for a single nanoparticle
```python
import os
import json
import pathlib
import numpy as np
from pydscat.dda import DDA

# GPU Device Config
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"

current_folder_path = pathlib.Path().absolute()

config = {'gpu_device': '/GPU:0',
          'dipole_length': 1,
          'min_wavelength': 0.4,
          'max_wavelength': 0.8,
          'num_wavelengths': 41,
          'ref_medium': 1.333,
          'rotation_steps': 10,
          'folder_path': None,
          'calculate_electricField': True,
          'lattice_constant': 0.41,
          'ref_data': [str(current_folder_path) + '/Au_ref_index.csv',str(current_folder_path) + '/Ag_ref_index.csv'],
          'metals': ["Au","Ag"],
          'dipole_data': str(current_folder_path)+ '/dipole_list.csv',
          "ratio":[1.0, 0.0],
          "method":"homo",
          "custom_ratio_path":None,
          'atom_data':None,
          'lattice_constant': None
        }
config['folder_path'] = str(current_folder_path)
np_dda = DDA(config)
np_dda.run_DDA()
np_dda.plot_spectra()
# Save the cross section data
np.savetxt(config['folder_path'] +"/data.csv",np.array(np_dda.C_cross_total),delimiter=",")
```

### [Example 2](./Examples/example2): simulation for a trajecotry
```python
# Import modules
import os
import json
import pathlib
import numpy as np
import pandas as pd
from pydscat.dda import DDA
from multiprocessing import Process
import tensorflow as tf
from matplotlib import pyplot as plt
import scipy.signal as signal
import scipy
def delete_1D(new_position,position):
    A=np.array(np.around(new_position,7)).tolist()
    B=np.array(np.around(position,7)).tolist()
    A = [i for i in A if i not in B]
    return A

# GPU Device Config
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"

current_folder_path = pathlib.Path().absolute()

# Define the overall dipole set 
position = []
for x in range(-30,30):
    for y in range(-30,30):
        for z in range(-30,30):
            if abs(x)+abs(y)+abs(z)<=20*np.sqrt(2)/2:
                position.append([x,y,z])
position = np.array(position)
np.savetxt('./data_Au@Ag_Octahedra/octahedra_overall.csv',position,delimiter=',')

# Define the core 
position_core = []
for x in range(-30,30):
    for y in range(-30,30):
        for z in range(-30,30):
            if abs(x)+abs(y)+abs(z)<=20*np.sqrt(2)/2*0.95:
                position_core.append([x,y,z])
position_core = np.array(position_core)
np.savetxt('./data_Au@Ag_Octahedra/octahedra_core.csv',position_core,delimiter=',')

# Define a trajecotry that randomly deletes the shell dipoles 
random_seed = 0
extra_dipole = np.array(delete_1D(position,position_core))
extra_index = [position.tolist().index(extra_dipole[i].tolist()) for i in range(len(extra_dipole))]
np.random.seed(random_seed)
np.random.shuffle(extra_index)
np.savetxt('./data_Au@Ag_Octahedra/random_delete_sequence_%d.csv'%random_seed,extra_index,delimiter=',')

# Generate the Au@Ag core shell structure
components = np.array([1.0,0.0]).reshape(-1,1).repeat(len(position),axis=1)
for i in extra_index:
    components[:,i] = np.array([0,1]).T # Define the dipole composed of Ag
np.savetxt(str(current_folder_path)+'/core_shell_components.csv',components,delimiter=',')

# Define the intial and final dipole sets
config = {'gpu_device': '/GPU:0',
            'dipole_length': 1.0,
            'min_wavelength': 0.4,
            'max_wavelength': 0.65,
            'num_wavelengths': 26,
            'ref_medium': 1.333,
            'rotation_steps': 10,
            'folder_path': None,
            'calculate_electricField': False,
            'ref_data': [str(current_folder_path) + '/Au_ref_index.csv',str(current_folder_path) + '/Ag_ref_index.csv'],
            'metals': ["Au","Ag"],
            'dipole_data': str(current_folder_path) + '/data_Au@Ag_Octahedra/octahedra_overall.csv',
            "lattice_constant":0.41,
            "method":"heter_custom",
            "custom_ratio_path":str(current_folder_path)+'/core_shell_components.csv',
            'atom_data': None,
            }
config['folder_path'] = str(current_folder_path) + '/data_Au@Ag_Octahedra/'
with open(config['folder_path']+'/config.json','w') as outfile:
    json.dump(config,outfile)

# Initialize the DDA simulation
np_dda = DDA(config)
alpha_j1 = np.array([np.repeat(np_dda.alpha_j[:,i],3) for i in range(np_dda.alpha_j.shape[1])]).T # Here we define the polarizibility of the original system
alpha_j2 = np.array([np.repeat(np_dda.alpha_j[:,i],3)/(10**10) for i in range(np_dda.alpha_j.shape[1])]).T # Here we define the polarizibility after deleting the dipoles
atom_index = [[i for i in range(3*j,3*j+3)] for j in extra_index] # the clarify sequence of deletion
atom_index = np.array(atom_index).flatten().tolist()
np_dda.flip_infor = [atom_index,alpha_j1,alpha_j2]

# Run the simulation
np_dda.calculate_spectrum_trajectories_v2()
data = np.array(np_dda.C_cross_total)
# Save the cross section data
np.savetxt(config['folder_path']+'/data.csv',data,delimiter=',')
```
