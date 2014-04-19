#!/bin/bash

for i in {1..64}
do
   sbt "run local[$i] data/kmeans_input_10000000.txt 5 0.01 output/workers$i.txt 10 1"
done
