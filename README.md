# WiggleJiggleMol
PDBXtra and Josico presents: Animating Quantum Mechanics Molecules... now in Blender
This repository provides a script to create, optimize, and animate molecular structures using Blender and various computational tools such as ASE, TorchANI, and NumPy.

Usage
A. Build Initial Non-Optimized Structure
A1. Creates the Balls for All Atom Positions

    Load the molecule from an XYZ file using ASE.
    Create a dictionary to map each atom to a Blender object (ball).
    Scale the balls based on predefined atom sizes.

A2. Create Bonds with Cylinders

    Generate possible bonds between atoms.
    Create cylinders to represent bonds and set their positions.

B. Add Axis to Move the Whole Molecule

    Add a mover object to allow moving the entire molecule within the Blender scene.

C. Color and Smooth the Balls
C1. Color

    Assign colors to atoms based on their element type.

C2. Smooth the Objects

    Smooth the balls and bonds in the Blender scene.

D. Geometry Optimization
D1. Do the Optimization

    Perform molecular geometry optimization using TorchANI and ASE.

D2. Do the Animation

    Animate the optimized molecule by updating atom positions over the frames.

E. Frequency Calculation
E1. Do the Frequency Calculation

    Calculate vibrational frequencies using TorchANI.

E2. Animate the Frequencies

    Animate the molecule to visualize the vibrational modes.

TODO

    Generalize the directory for pip installation.
    Add scaling for all periodic table elements.
    Add distances for making bonds appear or disappear based on length.
    Add colors for all the atoms.
    Improve implementation of dictionaries for atom-symbol and ball mappings.
    Implement a cutoff for bonds to optimize performance.
    Ask user if they want to perform geometry optimization.
    Add Psi4/PySCF for quantum mechanical optimization.
    Provide user options for animation trajectory length.
    Ask user for a pause between optimization and frequency animation.
    Allow user to select specific frequencies to display and number of wiggles.
