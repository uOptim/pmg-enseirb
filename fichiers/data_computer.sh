#!/bin/bash
# Run a simulation during a certain amount of time and print the average
# time for a step
CONF_FILE="bounce.conf"
RUN_TIME=5
FILE="execution_time"
# Launching atoms
./atoms --full-speed ${CONF_FILE} | egrep -o "[0-9]\.[0-9]+" >${FILE} &
# Letting atoms run during specified time
sleep ${RUN_TIME}
killall atoms
# Sleeping to let FILE be closed properly
sleep 0.1
# Computing average
cat ${FILE} | awk 'NR == 1 { sum=0 }; {sum+=$1;}; END {printf "%f\n", sum/NR}'
# Removing trace :
rm ${FILE}