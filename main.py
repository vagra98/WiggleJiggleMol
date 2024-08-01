#
# hola valentin
# 
import os
import sys
import time
import bpy
import numpy as np
import ase
import torch
import torchani

#import new directory to blender path
#TODO generalize the directory for the pip installation
functions_dir = "C:\\Users\\redbr\\Desktop\\blender_devving\\github\\"
if not functions_dir in sys.path:
    sys.path.append(functions_dir) 

#import functions
from functions import *

#constants (maybe TODO: add constants in another file and call it)
#TODO add scaling for all periodic table
atom_size_dict = {'H':0.2,'C':0.25,'O':0.4,'N':0.3}
#TODO add distances for making the bond appear or dissappear in function of the length
pt_dict = {
    'C': {'C': {'bond': 1.7}, 'H': {'bond': 1.2}, 'N': {'bond': 1.60}, 'O': {'bond': 1.42}},
    'H': {'C': {'bond': 1.2}, 'H': {'bond': 0.80}, 'N': {'bond': 1.1}, 'O': {'bond': 1.1}},
    'N': {'C': {'bond': 1.60}, 'H': {'bond': 1.1}, 'N': {'bond': 1.45}, 'O': {'bond': 1.42}},
    'O': {'C': {'bond': 1.42}, 'H': {'bond': 1.1}, 'N': {'bond': 1.42}, 'O': {'bond': 1.48}}
}
#TODO put colors for all the atoms 
#define color for each atom
element_colors = {'C': 'orange', 'H': 'white', 'N': 'blue', 'O': 'red'}
#create dictionary with the color and the Blender material object
#NOTE the first color is the name in script, the second color is the name shown in Blender; they are different
material_colors = {
    "green": bpy.data.materials.new("green"),
    "red": bpy.data.materials.new("red"),
    "grey": bpy.data.materials.new("grey"),
    "white": bpy.data.materials.new("white"),
    "blue": bpy.data.materials.new("blue"),
    "orange": bpy.data.materials.new("orange")
}
#add the color to the Blender material
material_colors["green"].diffuse_color = (0, 1, 0, 0.8)
material_colors["red"].diffuse_color = (1, 0, 0, 0.8)
material_colors["grey"].diffuse_color = (0.5, 0.5, 0.5, 0.8)
material_colors["white"].diffuse_color = (1, 1, 1, 0.8)
material_colors["blue"].diffuse_color = (0, 0, 1, 0.8)
material_colors["orange"].diffuse_color = (0.678, 0.341, 0.094, 1.0)


#A. Build initial non-optimized structure
start_create_initial_structure = time.time()

#read molecule with ase
#TODO user input for xyz file at his location
xyz_path = "C:\\Users\\redbr\\Desktop\\blender_devving\\github\\water.xyz.txt"
molecule = ase.io.read(xyz_path, format='xyz')      #load molecule in ase
atom_symbol = molecule.get_chemical_symbols()       #atom_symbol gets every chemical element in the molecule
number_of_atoms = len(atom_symbol)                  #number of atoms in the molecule

#A1. Creates the ballzzz for all atom positions; same size for all
#atomnum_bpyobj_dict is {atom_1}:{bpy object ball for atom 1}
atomnum_bpyobj_dict = atom_creation(molecule, number_of_atoms)

#bpyobj_atomsymb_dict is {bpy object ball for atom 1}:{atom_1} and changed {atom_1} to {H}
#TODO better implementation aka skill issue
bpyobj_atomsymb_dict = {value: key for key, value in atomnum_bpyobj_dict.items()}
bpyobj_atomsymb_dict.update(zip(bpyobj_atomsymb_dict.keys(), atom_symbol))

#do the scaling of the ballzzz
#bpyobj_atomscale_dict is {bpy object ball for atom 1}:{0.2}
bpyobj_atomscale_dict = bpyobj_atomsymb_dict.copy()
for key, value in bpyobj_atomscale_dict.items():
    if value in atom_size_dict:
        bpyobj_atomscale_dict[key] = atom_size_dict[value]
atom_scaling(bpyobj_atomscale_dict)

#A2. Create bonds with cylinders

#empty list of all the possible bonds
#TODO implement a cut-off, all bonds for 100 atoms molecule is shitty coding
all_bonds = []

#might be done with itertools library in order to avoid double loop
for elementA in range(number_of_atoms):
    for elementB in range(elementA + 1, number_of_atoms):
        all_bonds.append((elementA, elementB))

#empty list of all bonds lengths
long_size = np.array([])

#put cylinder for each bond
for index, bond in enumerate(all_bonds):
    #
    #atomA and atomB are the labels of the atoms to be bonded, they come from the tuple
    atomA, atomB = bond
    cylinder, long_size_val = cylinder_between(
        molecule.get_positions()[atomA][0], molecule.get_positions()[atomA][1], molecule.get_positions()[atomA][2],
        molecule.get_positions()[atomB][0], molecule.get_positions()[atomB][1], molecule.get_positions()[atomB][2], 
        0.1 )
    long_size = np.append(long_size, long_size_val)
    #TODO check if blender is in object mode already
    bpy.ops.object.mode_set(mode='OBJECT')
    #selects the bpy object and names it 
    cylinder = bpy.context.active_object
    cylinder.name = "bond_" + str(atomA) + "_" + str(atomB)



#B. Add the axis to move the whole molecule within the Blender scene
#TODO generalize "mover" name
mover = add_mover()
if mover.name == "mover1":
    mover.name = "molecule1"

#C. Color and smooth the balllzzz in Blender scene 
#TODO big: add two cylinders instead of one and color them properly
#C1. Color 
for object, element in bpyobj_atomsymb_dict.items():
    if element in element_colors:
        color_name = element_colors[element]
        if color_name in material_colors:
            material = material_colors[color_name]
            object.select_set(True)
            object.data.materials.append(material)
            object.select_set(False)

#C2. Smooth the 0=0 bby
for obj in bpy.data.objects:
    if "atom" in obj.name and obj.type == 'MESH':
        obj.select_set(True)
        toggle_shading(obj)
        obj.select_set(False)
    elif "bond" in obj.name and obj.type == 'MESH':
        obj.select_set(True)
        toggle_shading(obj)
        obj.select_set(False)

finish_create_initial_structure = time.time()
print("A. The total time to create the non-optimized structure is:")
print(finish_create_initial_structure - start_create_initial_structure )

#D. Geometry optimization
#TODO ask user if he wants a geometry optimization or not
#TODO add psi4/pyscf to do QM instead of ML optimization

start_optimization_structure = time.time()

#D1. Do the optimization
device = torch.device('cpu')
model = torchani.models.ANI1x(periodic_table_index=True).to(device).double()
calculator = model.ase()
molecule.set_calculator(calculator)
opt = ase.optimize.BFGS(molecule)
opt.run(fmax=1e-6)
total_opt_steps = opt.get_number_of_steps()

#D2. Do the animation 
#TODO ask user for larger or smaller animation trajectory; change total_opt_steps variable

#animate balllzzz
for index, atom in enumerate(atomnum_bpyobj_dict.keys()):
    atomnum_bpyobj_dict[atom].location =  molecule.get_positions()[index] 
    #1 second per optimization
    atomnum_bpyobj_dict[atom].keyframe_insert("location", frame=total_opt_steps)

#bake balllzzz
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.data.objects:
    if "atom" in obj.name and obj.type == 'MESH':
        obj.select_set(True)
bpy.ops.nla.bake(frame_start=1, frame_end=total_opt_steps, bake_types={'OBJECT'})


#follow balllzzz with the cylinders
cylinder_follow(molecule, all_bonds, total_opt_steps, long_size, pt_dict, atomnum_bpyobj_dict)

finish_optimization_structure = time.time()
print("A. The total time optimize the structure is:")
print(finish_optimization_structure - start_optimization_structure)

#E. Frequency calculation

start_frequency = time.time()

#E1. Do the frequency calculation 
#TODO ask for condition if the molecule is linear; there are 5 less degrees of freedom
species = torch.tensor(molecule.get_atomic_numbers(), device=device, dtype=torch.long).unsqueeze(0)
coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).requires_grad_(True)
masses = torchani.utils.get_atomic_masses(species)
energies = model((species, coordinates)).energies
hessian = torchani.utils.hessian(coordinates, energies=energies)
# The Hessian matrix should have shape `(1, 9, 9)`, where 1 means there is only# one molecule to compute, 9 means `3 atoms * 3D space = 9 degree of freedom`.
print(hessian.shape)
freq, modes, fconstants, rmasses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MDU')
torch.set_printoptions(precision=3, sci_mode=False)

print('Frequencies (cm^-1):', freq[6:])
print('Force Constants (mDyne/A):', fconstants[6:])
print('Reduced masses (AMU):', rmasses[6:])
print('Modes:', modes[6:])

normal_modes = np.zeros((3,3,3))
#go from torch to numpy
normal_modes = modes[6:].numpy()


#E2. Animate the frequencies


#the Blender frame where to start the frequency calculation
#TODO ask user the pause between the optimization and frequencies animation
pause_opt_freq = 20
start_freq = total_opt_steps + pause_opt_freq
#2 half_periods = 1 cycle; 1 cycle/24frames~ 1cycle/1sec animation
half_period = 12 #from one extreme position to the next
#how far should the frequency go until
amplitude = 1

#TODO big: ask which frequencies to display; default to the first 3 of smth
#TODO ask user how many wiggle jiggles want to do per frequency; use default 2 
#NOTE 2 is the first number, the second 2 is for the loop, fixed factor do not touch 
freq_count = 3
wiggles_jiggles_count = 2 * 2 
for freq_type in range(freq_count):  # change to how many frequencies and which ones does the user want
    #shift the starting frame for the second/third/... frequencies
    try:
        last_frame_freq_type
    except NameError:
        print('INITIAL FREQUENCY')
    else:
        print('SECOND THIRD FREQUENCY are declared')
        start_freq = last_frame_freq_type + pause_opt_freq
    for atom_number, (object, element) in enumerate(bpyobj_atomsymb_dict.items()):  # iterate over atoms in the molecule
        object.select_set(True)
        object.location = molecule.get_positions()[atom_number]  # optimized position, we start here
        object.keyframe_insert("location", frame=start_freq)
        for WJ in range(1, wiggles_jiggles_count, 2):
            print(WJ)
            min_frame = start_freq + (half_period * WJ)
            object.location = molecule.get_positions()[atom_number] - normal_modes[freq_type, atom_number, :] * amplitude  # first lower extreme
            print(min_frame)
            object.keyframe_insert("location", frame=min_frame)
            max_frame = start_freq + (half_period + half_period * WJ)
            object.location = molecule.get_positions()[atom_number] + normal_modes[freq_type, atom_number, :] * amplitude  # first upper extreme
            print(max_frame)
            object.keyframe_insert("location", frame=max_frame)
        #get back to optimized position
    last_frame_freq_type = max_frame + half_period
    object.location = molecule.get_positions()[atom_number]  # optimized position, we start here
    object.keyframe_insert("location", frame=last_frame_freq_type)
    object.select_set(False)


finish_frequency = time.time()
print("A. The total time to get frequencies is:")
print(finish_frequency - start_frequency)
