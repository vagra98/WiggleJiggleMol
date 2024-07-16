import bpy
import numpy as np
from ase.io.cube import read_cube
import skimage as ski

cube_path = "I:\\Orca\\ejemplo_cube_gen\\h2o.inp.mo4a.cube"
cube_file = open(cube_path, "r")

 #atoms, data, origin = ase.io.cube.read_cube(cube_file, read_data=True, program=None, verbose=False)
cube_dict = read_cube(cube_file, read_data=True, program=None, verbose=False)
print("cell is 3x3")
print(cube_dict["atoms"].cell)
#print(cube_dict)
#print(cube_dict['data'])

volume = np.array(cube_dict['data'])
#print(volume)
print(volume.shape)
print("volume max", volume.max())
print("volume min", volume.min())


thresh = -0.05
vertices, faces, normals, values = ski.measure.marching_cubes(volume, level=thresh, method="lorensen")

print(vertices[:10])
print(faces[:10])

mesh = bpy.data.meshes.new(name="try")
#mesh.from_pydata(verts, edges, faces)
mesh.from_pydata(vertices,[],faces)


locO1 = np.array((0.0,0.0,0.118297)) + 20 
locH1 = np.array((-1.496663,    0.000000,   -0.939761)) + 20
locH2 = np.array((1.496663,    0.000000,   -0.939761)) + 20

bpy.ops.mesh.primitive_ico_sphere_add(radius=1, enter_editmode=False, align='WORLD', location=locO1, scale=(1, 1, 1))
bpy.ops.mesh.primitive_ico_sphere_add(radius=1, enter_editmode=False, align='WORLD', location=locH1, scale=(1, 1, 1))
bpy.ops.mesh.primitive_ico_sphere_add(radius=1, enter_editmode=False, align='WORLD', location=locH2, scale=(1, 1, 1))

