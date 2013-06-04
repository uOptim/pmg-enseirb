set title "Comparaison des performances entres différentes taille de slices pour la troisième version du noyau collision"

set xlabel "Nombre d'atomes"
set ylabel "Temps d'exécution moyen du kernel (en ms)"

set tics out

set terminal png size 1024,768
set output "figures/collisions_slices.png"

plot '../results/collisions_v3.dat'     with lines title "Slice 16",\
     '../results/collisions_v3_64.dat'  with lines title "Slice 64",\
     '../results/collisions_v3_256.dat' with lines title "Slice 256"
