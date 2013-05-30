#!/bin/bash

KINIT=50
KMAX=2000

#run_simulation(dest_file, params
function run_simulation {
    k=$KINIT
    # TODO review condition
    while [ $k -lt $KMAX ]
    do
        AVG_TIME=$(./data_computer.sh $k.conf -lf ${@:2} )
        echo $k $AVG_TIME
        k=$(( $k * 2))
    done
}

k=$KINIT
#TODO generate python files
while [ $k -lt $KMAX ]
do
    python3 file_generator.py $k > $k.conf
    k=$(( $k * 2))    
done


run_simulation