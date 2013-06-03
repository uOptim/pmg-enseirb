#!/bin/bash

# Run the simulation from init to max by multipling k nb_atoms by n
# at each step
# data_builder.sh <init> <max> <step_mult> options

KINIT=$1
KMAX=$2
STEP_MULT=$3

PARAMETERS=${@:4}

#run_simulation(params)
function run_simulation {
    k=$KINIT
    # TODO review condition
    while [ $k -lt $KMAX ]
    do
        AVG_TIME=$(./data_computer.sh $k.conf $@ )
        echo $k $AVG_TIME
        k=$(( $k + $STEP_MULT))
    done
}

k=$KINIT
#TODO generate python files
while [ $k -lt $KMAX ]
do
    if [ ! -f $k.conf ]
    then
        python3 file_generator.py $k > $k.conf
    fi
    k=$(( $k + $STEP_MULT))
done


run_simulation ${PARAMETERS}
