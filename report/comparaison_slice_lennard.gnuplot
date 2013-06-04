set title "Comparaison des performances entres différentes taille de slices pour la seconde version du noyau Lennard-Jones"

set xlabel "Nombre d'atomes"
set ylabel "Temps d'exécution moyen du kernel (en ms)"

set tics out

set terminal png size 1024,768
set output "figures/lennard_slices.png"

plot '../results/lennard_v2_16.dat'  with linespoints title "Slice 16",\
     '../results/lennard_v2_64.dat'  with linespoints title "Slice 64",\
     '../results/lennard_v2_256.dat' with linespoints title "Slice 256"
