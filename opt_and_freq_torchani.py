import bpy
import bmesh
import numpy as np
import ase
import ase.optimize
from ase.geometry.analysis import Analysis
import torch
import torchani
import math


#PART 1: SETUP INITIAL STRUCTURE AND CALCULATION TYPE
###############################################################################
# Let's now manually specify the device we want TorchANI to run:
#set optimization to CPU
device = torch.device('cpu')
#choose model to optimize, ANI1x in this case; see more of ASE/TorchaANI for better calculations
model = torchani.models.ANI1x(periodic_table_index=True).to(device).double()

###############################################################################
# Let's first construct a water molecule and do structure optimization:
#put the path of the xyz file
#METHANE
xyz_path = "C:\\Users\\redbr\\Desktop\\blender_devving\\water.xyz.txt"
#TFG ZIKA
#xyz_path = "C:\\Users\\redbr\\Desktop\\blender_devving\\L-Arginine.xyz"
#read file in variable molecule; will be used for every ase command
molecule = ase.io.read(xyz_path, format='xyz')
#set the calculator to ase
calculator = model.ase()
molecule.set_calculator(calculator) 
print("molecule variable is:")
print(molecule)

#insert unoptimized molecules in blender
#stores in order the chemical symbols, e.g. CH4 C,H,H,H,H
atom_names = molecule.get_chemical_symbols()
number_of_atoms = len(atom_names)
#collects number of atoms
print("number of atoms")
print(number_of_atoms)

#atom_creation is the function that puts the meshes for the atoms of the molecules; bonds will be done later
atoms_dic = {}
def atom_creation(number_of_atoms):
    atoms_dic = {}
    for i in range(number_of_atoms):
        atom_name = "atom_" + str(i)
        atoms_dic[atom_name] = bpy.ops.mesh.primitive_ico_sphere_add(radius=1, enter_editmode=False, align='WORLD', location=(molecule.get_positions()[i]), scale=(1, 1, 1),subdivisions=4)
        atoms_dic[atom_name] = bpy.context.active_object
        atoms_dic[atom_name].name = "atom_" + str(i)
        #TODO ask user if he wants to see the animation of the optimization
        atoms_dic[atom_name].keyframe_insert("location", frame=1)
        print("ATOM icosphere created successfully")
        print(atoms_dic[atom_name])
    return atoms_dic
atoms_dic = atom_creation(number_of_atoms)

print("ATOMS DICTIONARY IS")
print(atoms_dic)
#por ejemplo {'atom_0': bpy.data.objects['atom_0'], 'atom_1': bpy.data.objects['atom_1'], 'atom_2': bpy.data.objects['atom_2']}
#PART 2: change scale of atoms; update for all atoms in PT

#change scale in function of atom type; reverse the dictionary
reverse_atoms_dic = {value: key for key, value in atoms_dic.items()}

#convert dictionary values to a list
values_list = list(reverse_atoms_dic.values())

#substitute the third value with a string
for i in range(number_of_atoms):
    values_list[i] = molecule.get_chemical_symbols()[i]

    
# Update the dictionary with modified values
reverse_atoms_dic.update(zip(reverse_atoms_dic.keys(), values_list))

print("Chemical species per object")
print(reverse_atoms_dic)

object_element_dic = reverse_atoms_dic.copy()

size_atoms_dic = reverse_atoms_dic.copy()

atom_size_dic = {'H':0.2,'C':0.25,'O':0.4,'N':0.3}

for key, value in size_atoms_dic.items():
    if value in atom_size_dic:
        size_atoms_dic[key] = atom_size_dic[value]
print("Size dictionary")
print(size_atoms_dic)

def atom_scaling(size_atoms_dic):
    for key,value in size_atoms_dic.items():
        #atom = key
        key.scale = (value, value, value)
        print("key is/value is")
        print(key, value)
        #atoms_dic[atom_name].name = "atom_" + str(i)
        #atoms_dic[atom_name].keyframe_insert("location", frame=1)
        print("ATOM icosphere scaled successfully")
        #print(atoms_dic[atom_name])
    return

atom_scaling(size_atoms_dic)

# enlaces:

allbonds = []
ana = Analysis(molecule)
for elementA in list(set(molecule.get_chemical_symbols())):
    for elementB in list(set(molecule.get_chemical_symbols())):
        if (elementA != elementB):
            CHBonds = ana.get_bonds(elementA, elementB, unique=True)
            print("CH Bonds are")
            print(CHBonds)
            allbonds.extend(CHBonds)
# Create an empty set to store unique pairs
unique_pairs = set()

# Iterate through the original list and add unique pairs to the set
for sublist in allbonds:
     for pair in sublist:
        sorted_pair = tuple(sorted(pair))
        # Check if the pair is already in the set
        if pair not in unique_pairs:
            unique_pairs.add(sorted_pair)
    #unique_pairs.update(sublist)

# Convert the set back to a list if necessary
unique_bonds = list(unique_pairs)

print("Unique Bonds")
print(unique_bonds)

#CH Bonds are
#[[(0, 1), (0, 2)]]
#CH Bonds are
#[[(1, 0), (2, 0)]]
#Unique Bonds
#[(0, 1), (0, 2)]
#ALL BONDS
#[(0, 1), (0, 2), (1, 2)]

def bond_creator(start_point,end_point):
    # Define the Cartesian coordinates of the start and end points
    #start_point = molecule.get_positions()[1]  # Replace with your start point coordinates
    #end_point = molecule.get_positions()[2]    # Replace with your end point coordinates


    # Append a scalar to each coordinate in start_point and end_point
    scalar = 1  # Example scalar value, replace with your desired value
    start_point = np.append(start_point, scalar)
    end_point = np.append(end_point, scalar)

    # Create a new curve object
    curve_data = bpy.data.curves.new('PathCurve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 2
    curve_data.bevel_depth = 0.02


    # Create a new spline in the curve data
    spline = curve_data.splines.new(type='POLY')
    spline.points.add(1)  # Add 1 point to the spline
    
    # Set the coordinates of the start and end points
    spline.points[0].co = start_point.tolist()
    spline.points[1].co = end_point.tolist()

    # Create a new object with the curve data
    curve_obj = bpy.data.objects.new('PathCurveObject', curve_data)
    #curve_obj.keyframe_insert(data_path="location", frame=keyframe)
    #curve_data.keyframe_insert("location", frame=1)
    bpy.context.collection.objects.link(curve_obj)
    

    # Set the object's origin to the start point
    #curve_obj.location = start_point[:3].tolist()
    return()
#bond_creator(molecule.get_positions()[1],molecule.get_positions()[2])
#bond_creator(molecule.get_positions()[0],molecule.get_positions()[1])
#bond_creator(molecule.get_positions()[0],molecule.get_positions()[2])
#bond_creator(molecule.get_positions()[0],molecule.get_positions()[3])
#bond_creator(molecule.get_positions()[0],molecule.get_positions()[4])




def cylinder_between(x1, y1, z1, x2, y2, z2, r):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1    
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    cylinder = bpy.ops.mesh.primitive_cylinder_add(
        radius = r, 
        depth = dist,
        location = (dx/2 + x1, dy/2 + y1, dz/2 + z1)
    ) 

    #phi = math.atan2(dy, dx) 
    #theta = math.acos(dz/dist) 

    #bpy.context.object.rotation_euler[1] = theta 
    #bpy.context.object.rotation_euler[2] = phi 
    return(cylinder,dist)
#cyli = cylinder_between(molecule.get_positions()[0][0],molecule.get_positions()[0][1],molecule.get_positions()[0][2],molecule.get_positions()[3][0],molecule.get_positions()[3][1],molecule.get_positions()[3][2],0.05)

#1. Add empty object at atomA


#how to add a hook; delete later
#hookC = bpy.ops.object.empty_add(location=molecule.get_positions()[0])
#hookC = bpy.context.active_object
#hookC.name = "hookC"



#long_size = np.zeros(len(unique_bonds))
#i=-1
#for bond in unique_bonds:
#    i +=1
#    atomA, atomB = bond
#    print("AtomA", atomA, "AtomB", atomB)
#    cylinder, long_size[i] = cylinder_between(molecule.get_positions()[atomA][0],molecule.get_positions()[atomA][1],molecule.get_positions()[atomA][2],molecule.get_positions()[atomB][0],molecule.get_positions()[atomB][1],molecule.get_positions()[atomB][2],0.05)
#    bpy.ops.object.mode_set(mode = 'OBJECT')
#    cylinder = bpy.context.active_object
#    cylinder.name = "bond" + str(atomA) + str(atomB)
    
#create tuple for all atom pairs
#TODO cambiar a dict
all_bonds = []
for elementA in range(number_of_atoms):
    for elementB in range(elementA+1,number_of_atoms):
        pair_tuple = (elementA, elementB)
        all_bonds.append(pair_tuple)
print("ALL BONDS")
print(all_bonds)
long_size = np.zeros(len(all_bonds))
i = -1
for bond in all_bonds:
    i +=1
    atomA, atomB = bond
    print("AtomA", atomA, "AtomB", atomB)
    cylinder, long_size[i] = cylinder_between(molecule.get_positions()[atomA][0],molecule.get_positions()[atomA][1],molecule.get_positions()[atomA][2],molecule.get_positions()[atomB][0],molecule.get_positions()[atomB][1],molecule.get_positions()[atomB][2],0.1)
    bpy.ops.object.mode_set(mode = 'OBJECT')
    cylinder = bpy.context.active_object
    cylinder.name = "bond" + str(atomA) + str(atomB)



#C = bpy.context

# Reference two cylinder objects
#c1 = bpy.data.objects['atom_0']
#c2 = bpy.data.objects['atom_3']


#m = bpy.data.meshes.new('connector')

#bm = bmesh.new()
#v1 = bm.verts.new(molecule.get_positions()[0])
#v2 = bm.verts.new(molecule.get_positions()[3])
#e  = bm.edges.new([v1,v2])

#bm.to_mesh(m)

#o = bpy.data.objects.new( 'connector', m )
#C.scene.collection.objects.link( o )

# Hook connector vertices to respective cylinders
#for i, cyl in enumerate([ c1, c2 ]):
#    bpy.ops.object.select_all( action = 'DESELECT' )
#    cyl.select_set(True)
#    o.select_set(True)
#    C.view_layer.objects.active = o # Set connector as active

    # Select vertex
#    bpy.ops.object.mode_set(mode='OBJECT')
#    o.data.vertices[i].select = True    
#    bpy.ops.object.mode_set(mode='EDIT')

#    bpy.ops.object.hook_add_selob() # Hook to cylinder

#    bpy.ops.object.mode_set(mode='OBJECT')
#    o.data.vertices[i].select = False 

#o.modifiers.new('Skin', 'SKIN')
#bpy.ops.object.select_all( action = 'DESELECT' )





#d = 1.2
#t = math.pi / 180 * 104.51
#molecule = ase.Atoms('H2O', positions=[
#    (d, 0, 0),
#    (d * math.cos(t), d * math.sin(t), 0),
#    (0, 0, 0),
#], calculator=model.ase())
print("\n NOT Optimized positions:")
print(molecule.get_positions())
opt = ase.optimize.BFGS(molecule)
opt.run(fmax=1e-6)




total_opt_steps = opt.get_number_of_steps()

#for i in range(number_of_atoms):
#    atom_name = "atom_" + str(i)
#    atoms_dic[atom_name].location =  molecule.get_positions()[i] #normal_modes[0,2,:]
    #1 second per optimization
    #atoms_dic[atom_name].keyframe_insert("location", frame=24)

#frame_start = bpy.context.scene.frame_start
#frame_end = bpy.context.scene.frame_end


#TODO put real values and complete the whole PT
pt_dic = {
    'C': {
        'C': {'bond': 1.7},
        'H': {'bond': 1.2},
        'N': {'bond': 1.60},
        'O': {'bond': 1.42}
    },
    'H': {
        'C': {'bond': 1.2},
        'H': {'bond': 0.80},
        'N': {'bond': 1.1},
        'O': {'bond': 1.1}
    },
    'N': {
        'C': {'bond': 1.60},
        'H': {'bond': 1.1},
        'N': {'bond': 1.45},
        'O': {'bond': 1.42}
    },
    'O': {
        'C': {'bond': 1.42},
        'H': {'bond': 1.1},
        'N': {'bond': 1.42},
        'O': {'bond': 1.48}
    }
}


frame_start = 1
frame_end = total_opt_steps



bpy.ops.object.select_all(action='DESELECT')


for obj in bpy.data.objects:
        if "atom" in obj.name and obj.type == 'MESH':
            obj.select_set(True)
bpy.ops.nla.bake(frame_start=frame_start, frame_end=frame_end, bake_types={'OBJECT'})


#Variable description:
#all_bonds
def cylinder_follow(molecule, all_bonds, total_opt_steps, long_size, pt_dic):
    j=-1
    for bond in all_bonds:
        j += 1
        atomA, atomB = bond
        #print("atomA", atomA, "name", molecule.get_chemical_symbols()[atomA], "atomB", atomB, "name", molecule.get_chemical_symbols()[atomB])
        atom_nameA = "atom_" + str(atomA)
        atom_nameB = "atom_" + str(atomB)
        cylinder_name = "bond" + str(atomA) + str(atomB)
        for i in range(1,total_opt_steps):
            bpy.context.scene.frame_set(i)
            #dx = molecule.get_positions()[3][0] - molecule.get_positions()[0][0]
            #dy = molecule.get_positions()[3][1] - molecule.get_positions()[0][1]
            #dz = molecule.get_positions()[3][2] - molecule.get_positions()[0][2]
            dx = atoms_dic[atom_nameB].location[0] - atoms_dic[atom_nameA].location[0]
            dy = atoms_dic[atom_nameB].location[1] - atoms_dic[atom_nameA].location[1]
            dz = atoms_dic[atom_nameB].location[2] - atoms_dic[atom_nameA].location[2]
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            scaling_long = dist/long_size[j]
            cyl03 = bpy.data.objects[cylinder_name]
            #cyl03.scale[2] = scaling_long
            #cyl03.keyframe_insert("scale", frame=i)
            cyl03.location = (dx/2 + atoms_dic[atom_nameA].location[0], dy/2 + atoms_dic[atom_nameA].location[1], dz/2 + atoms_dic[atom_nameA].location[2])
       
            #cylinder = bpy.ops.mesh.primitive_cylinder_add(
            #    radius = r, 
            #    depth = dist,
            #    location = (dx/2 + x1, dy/2 + y1, dz/2 + z1)   
            #) 

            phi = math.atan2(dy, dx) 
            theta = math.acos(dz/dist) 

            #bpy.context.object.rotation_euler[1] = theta 
            #bpy.context.object.rotation_euler[2] = phi 
            cyl03.rotation_euler[1] = theta 
            cyl03.rotation_euler[2] = phi
            print("name", molecule.get_chemical_symbols()[atomA], "name", molecule.get_chemical_symbols()[atomB], "distance", dist, "decision", pt_dic[molecule.get_chemical_symbols()[atomA]][molecule.get_chemical_symbols()[atomB]]['bond'])
            cyl03.keyframe_insert("rotation_euler", frame=i) 
            cyl03.keyframe_insert("location", frame=i)
            print()
            print()
            print("FRAME NUMBER: ", i)
            print("atom A", atomA, "atom B", atomB)
            #if (cyl03.dimensions.z < pt_dic[molecule.get_chemical_symbols()[atomA]][molecule.get_chemical_symbols()[atomB]]['bond']):
            if (dist > pt_dic[molecule.get_chemical_symbols()[atomA]][molecule.get_chemical_symbols()[atomB]]['bond']):
                print("should be deleted")
                #cyl03.scale[2] = scaling_long
                cyl03.scale = (0.0, 0.0, 0.0)
                print(dist, " > ", pt_dic[molecule.get_chemical_symbols()[atomA]][molecule.get_chemical_symbols()[atomB]]['bond'])
                cyl03.keyframe_insert("scale", frame=i)
            else:
                print("should be kept")
                print(dist, " < ", pt_dic[molecule.get_chemical_symbols()[atomA]][molecule.get_chemical_symbols()[atomB]]['bond'])
                cyl03.scale[0] = 1.0
                cyl03.scale[1] = 1.0
                cyl03.scale[2] = scaling_long
                #cyl03.scale = (0.0, 0.0, 0.0)
                cyl03.keyframe_insert("scale", frame=i)
            cyl03.scale[0] = 1.0
            cyl03.scale[1] = 1.0
            cyl03.scale[2] = scaling_long
    return()
#cylinder_follow(molecule, all_bonds, total_opt_steps, long_size, pt_dic)    
        
for i in range(1,24):
    bpy.context.scene.frame_set(i)
    for elementA in range(number_of_atoms):
        for elementB in range(number_of_atoms):
            if (elementA != elementB):
                #print("Element A - Element B distance:")
                bond_length = molecule.get_distance(elementA,elementB)
                #print(bond_length)


Bonds = ana.unique_bonds
print("should work huh")
print(Bonds)

#c1.location = c1_fin.location
#c1.keyframe_insert("location", frame=30)
#bond_creator(molecule.get_positions()[0],molecule.get_positions()[1])
#bond_creator(molecule.get_positions()[0],molecule.get_positions()[2])




###############################################################################
# Now let's extract coordinates and species from ASE to use it directly with
# TorchANI:
species = torch.tensor(molecule.get_atomic_numbers(), device=device, dtype=torch.long).unsqueeze(0)
coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).requires_grad_(True)

###############################################################################
# TorchANI needs the masses of elements in AMU to compute vibrations. The
# masses in AMU can be obtained from a tensor with atomic numbers by using
# this utility:
masses = torchani.utils.get_atomic_masses(species)

###############################################################################
# To do vibration analysis, we first need to generate a graph that computes
# energies from species and coordinates. The code to generate a graph of energy
# is the same as the code to compute energy:
energies = model((species, coordinates)).energies

###############################################################################
# We can now use the energy graph to compute analytical Hessian matrix:
hessian = torchani.utils.hessian(coordinates, energies=energies)

###############################################################################
# The Hessian matrix should have shape `(1, 9, 9)`, where 1 means there is only# one molecule to compute, 9 means `3 atoms * 3D space = 9 degree of freedom`.
print(hessian.shape)

###############################################################################
# We are now ready to compute vibrational frequencies. The output has unit
# cm^-1. Since there are in total 9 degree of freedom, there are in total 9
# frequencies. Only the frequencies of the 3 vibrational modes are interesting.
# We output the modes as MDU (mass deweighted unnormalized), to compare with ASE.
freq, modes, fconstants, rmasses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MDU')
torch.set_printoptions(precision=3, sci_mode=False)

print('Frequencies (cm^-1):', freq[6:])
print('Force Constants (mDyne/A):', fconstants[6:])
print('Reduced masses (AMU):', rmasses[6:])
print('Modes:', modes[6:])


print("\nOptimized positions:")
print(molecule.get_positions()[0])
normal_modes = np.zeros((3,3,3))
for i in range(6,9):
    print(torch.Tensor.size(modes[i]))
    #normal_modes[i,:,:] = modes[i].numpy()
#normal_modes = torch.numpy(modes[6:])
normal_modes = modes[6:].numpy()
print(normal_modes[0,0,:])


def add_mover():
    mover = bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    mover = bpy.context.active_object
    mover.name = "mover1"
    for obj in bpy.data.objects:
        if "atom" in obj.name and obj.type == 'MESH':
            obj.select_set(True)
        elif "bond" in obj.name and obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.parent_set(type='OBJECT')
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    return(mover)
mover = add_mover()
#TODO change the molecule so more can be added and it's fine
if mover.name == "mover1":
    mover.name = "molecule1"
    
    
    
def set_shading(object, OnOff=True):
    """ Set the shading mode of an object
        True means turn smooth shading on.
        False means turn smooth shading off.
    """
    if not object:
        return
    if not object.type == 'MESH':
        return
    if not object.data:
        return
    polygons = object.data.polygons
    polygons.foreach_set('use_smooth',  [OnOff] * len(polygons))
    object.data.update()

def toggle_shading(object):
    """ Toggle the shading mode of an object """
    if not object:
        return
    if not object.type == 'MESH':
        return
    if not object.data:
        return
    polygons = object.data.polygons
    for polygon in polygons:
        polygon.use_smooth = not polygon.use_smooth
    object.data.update()

for obj in bpy.data.objects:
    if "atom" in obj.name and obj.type == 'MESH':
        obj.select_set(True)
        toggle_shading(obj)
        obj.select_set(False)
    elif "bond" in obj.name and obj.type == 'MESH':
        obj.select_set(True)
        toggle_shading(obj)
        obj.select_set(False)
print(atoms_dic)     
        

element_colors = {
    'C': 'orange',   # Grey color for carbon
    'H': 'white',  # White color for hydrogen
    'N': 'blue',   # Blue color for nitrogen
    'O': 'red'     # Red color for oxygen
}


mat_green = bpy.data.materials.new("green")
mat_green.diffuse_color = (0,1,0,0.8)
mat_red = bpy.data.materials.new("red")
mat_red.diffuse_color = (1,0,0,0.8)
mat_grey = bpy.data.materials.new("grey")
mat_grey.diffuse_color = (0.5,0.5,0.5,0.8)  
mat_white = bpy.data.materials.new("white")
mat_white.diffuse_color = (1,1,1,0.8)       
mat_blue = bpy.data.materials.new("blue")
mat_blue.diffuse_color = (0,0,1,0.8)
mat_orange = bpy.data.materials.new("orange")
mat_orange.diffuse_color = (0.6784313725490196, 0.3411764705882353, 0.09411764705882353, 1.0)

material_colors = {
    "green": mat_green,
    "red": mat_red,
    "grey": mat_grey,
    "white": mat_white,
    "blue": mat_blue,
    "orange": mat_orange
}


for object, element in object_element_dic.items():
    if element in element_colors:
        color_name = element_colors[element]
        if color_name in material_colors:
            material = material_colors[color_name]
            object.select_set(True)
            print(object)
            object.data.materials.append(material)
            object.select_set(False)
    
print(object_element_dic)
print(element_colors)
print(material_colors)



#O1 = bpy.context.active_object
#O1.name = "atom3"
#O1.keyframe_insert("location", frame=1)
##normal_modes[number_mode,atom,xyz]
#O1.location = molecule.get_positions()[2] + normal_modes[0,2,:]
#O1.keyframe_insert("location", frame=30)
#O1.location =  molecule.get_positions()[2] - normal_modes[0,2,:]
#O1.keyframe_insert("location", frame=60)
#O1.location =  molecule.get_positions()[2] + normal_modes[0,2,:]
#O1.keyframe_insert("location", frame=90)
#O1.location =  molecule.get_positions()[2] - normal_modes[0,2,:]
#O1.keyframe_insert("location", frame=120)
#O1.location =  molecule.get_positions()[2]
#O1.keyframe_insert("location", frame=150)


#start from total opt steps + 12 (half second at 24fps)
start_freq = total_opt_steps 
half_period = 12 #from one extreme position to the next
amplitude = 1
last_frame_freq_type = 19 
freq_count = 3*len(molecule) -6


for freq_type in range(freq_count): #change to 3N-6
    atom_number = -1
    start_freq = last_frame_freq_type
    for object, element in object_element_dic.items():
        atom_number += 1 #iterate over atoms in the molecule
        object.select_set(True)
        object.location = molecule.get_positions()[atom_number] #optimized position, we start here
        object.keyframe_insert("location", frame=start_freq)
        object.location = molecule.get_positions()[atom_number] - normal_modes[freq_type,atom_number,:] * amplitude #first lower extreme
        object.keyframe_insert("location", frame=start_freq+12)
        object.location = molecule.get_positions()[atom_number] + normal_modes[freq_type,atom_number,:] * amplitude #first upper extreme
        object.keyframe_insert("location", frame=start_freq+24)
        object.location = molecule.get_positions()[atom_number] - normal_modes[freq_type,atom_number,:] * amplitude #first lower extreme
        object.keyframe_insert("location", frame=start_freq+36)
        object.location = molecule.get_positions()[atom_number] + normal_modes[freq_type,atom_number,:] * amplitude #first upper extreme
        object.keyframe_insert("location", frame=start_freq+48)
        object.location = molecule.get_positions()[atom_number] #optimized position, we start here
        object.keyframe_insert("location", frame=start_freq+60)
        last_frame_freq_type = total_opt_steps + (freq_type+1)*60
        object.select_set(False)

cylinder_follow(molecule, all_bonds, last_frame_freq_type, long_size, pt_dic)    