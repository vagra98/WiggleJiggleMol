import bpy
import bmesh
import numpy as np
import ase
import ase.optimize
from ase.geometry.analysis import Analysis
import torch
import torchani

def atom_creation(molecule, number_of_atoms):
    atoms_dic = {}
    for i in range(number_of_atoms):
        atom_name = "atom_" + str(i)
        atoms_dic[atom_name] = bpy.ops.mesh.primitive_ico_sphere_add(
            radius=1, 
            enter_editmode=False, 
            align='WORLD', 
            location=(molecule.get_positions()[i]), 
            scale=(1, 1, 1),
            subdivisions=4
        )
        atoms_dic[atom_name] = bpy.context.active_object
        atoms_dic[atom_name].name = "atom_" + str(i)
        atoms_dic[atom_name].keyframe_insert("location", frame=1)
    return atoms_dic

def atom_scaling(size_atoms_dic):
    for key, value in size_atoms_dic.items():
        key.scale = (value, value, value)
    return

def bond_creator(molecule, start_point, end_point):
    scalar = 1
    start_point = np.append(start_point, scalar)
    end_point = np.append(end_point, scalar)

    curve_data = bpy.data.curves.new('PathCurve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 2
    curve_data.bevel_depth = 0.02

    spline = curve_data.splines.new(type='POLY')
    spline.points.add(1)
    spline.points[0].co = start_point.tolist()
    spline.points[1].co = end_point.tolist()

    curve_obj = bpy.data.objects.new('PathCurveObject', curve_data)
    bpy.context.collection.objects.link(curve_obj)
    return

def cylinder_between(x1, y1, z1, x2, y2, z2, r):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    cylinder = bpy.ops.mesh.primitive_cylinder_add(
        radius=r,
        depth=dist,
        location=(dx/2 + x1, dy/2 + y1, dz/2 + z1)
    )
    return cylinder, dist

def cylinder_follow(molecule, all_bonds, total_opt_steps, long_size, pt_dic, atoms_dic):
    j = -1
    for bond in all_bonds:
        j += 1
        atomA, atomB = bond
        atom_nameA = "atom_" + str(atomA)
        atom_nameB = "atom_" + str(atomB)
        cylinder_name = "bond_" + str(atomA) + "_" + str(atomB)
        for i in range(1, total_opt_steps):
            bpy.context.scene.frame_set(i)
            half_dist_x = (atoms_dic[atom_nameB].location[0] + atoms_dic[atom_nameA].location[0]) / 2
            half_dist_y = (atoms_dic[atom_nameB].location[1] + atoms_dic[atom_nameA].location[1]) / 2
            half_dist_z = (atoms_dic[atom_nameB].location[2] + atoms_dic[atom_nameA].location[2]) / 2
            dx = half_dist_x - atoms_dic[atom_nameA].location[0]
            dy = half_dist_y - atoms_dic[atom_nameA].location[1]
            dz = half_dist_z - atoms_dic[atom_nameA].location[2]
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            scaling_long = dist / long_size[j]
            cyl03 = bpy.data.objects[cylinder_name]
            cyl03.location = (dx / 2 + atoms_dic[atom_nameA].location[0], dy / 2 + atoms_dic[atom_nameA].location[1], dz / 2 + atoms_dic[atom_nameA].location[2])

            phi = np.arctan2(dy, dx)
            theta = np.arccos(dz / dist)

            cyl03.rotation_euler[1] = theta
            cyl03.rotation_euler[2] = phi
            cyl03.keyframe_insert("rotation_euler", frame=i)
            cyl03.keyframe_insert("location", frame=i)
            if dist > pt_dic[molecule.get_chemical_symbols()[atomA]][molecule.get_chemical_symbols()[atomB]]['bond']:
                cyl03.scale = (0.0, 0.0, 0.0)
                cyl03.keyframe_insert("scale", frame=i)
            else:
                cyl03.scale = (1.0, 1.0, scaling_long)
                cyl03.keyframe_insert("scale", frame=i)
        cylinder_name = "bond_" + str(atomA) + "_" + str(atomB)
    return

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
    return mover

def set_shading(object, OnOff=True):
    if not object:
        return
    if not object.type == 'MESH':
        return
    if not object.data:
        return
    polygons = object.data.polygons
    polygons.foreach_set('use_smooth', [OnOff] * len(polygons))
    object.data.update()

def toggle_shading(object):
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
    return
