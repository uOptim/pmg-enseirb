set title "Comparaison des performances entres diverses implémentations du noyau collision"

set xlabel "Nombre d'atomes"
set ylabel "Temps d'exécution moyen du kernel"

set tics out

set terminal png size 1024,768
set output "figures/collision.png"

plot '../results/collisions_v1.dat' with lines title "Collision v1", \
     '../results/collisions_v2.dat' with lines title "Collision v2", \
     '../results/collisions_v3.dat' with lines title "Collision v3"  