set title "Comparaison des performances entres diverses implémentations du noyau Lennard-Jones"

set xlabel "Nombre d'atomes"
set ylabel "Temps d'exécution moyen du kernel"

set tics out

set terminal png size 1024,768
set output "figures/lennard_versions.png"

plot '../results/lennard_v1.dat'    with lines title "Lennard-Jones v1",\
     '../results/lennard_v2.dat' with lines title "Lennard-Jones v2"