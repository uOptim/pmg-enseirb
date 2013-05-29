import sys
import random

# Handling parameters
nb_atoms = int(sys.argv[1])

coord_min = 0.0
coord_max = 10000.0
speed_min = 0.0
speed_max = 0.5

# Generating atoms
atoms = []
for i in range(nb_atoms):
    p = []
    v = []
    for j in range(3):
        p += [random.uniform(coord_min, coord_max)]
        v += [random.uniform(speed_min, speed_max)]
    atoms += [[p,v]]

# Printing header
print(len(atoms))
for i in range(3):
    print(coord_min, coord_max);
print(1)
    

# Printing atoms
for i in range(nb_atoms):
    for j in range(3):
        print(atoms[i][0][j], end=' ')
    print("")
    for j in range(3):
        print(atoms[i][1][j], end=' ')
    print("")
