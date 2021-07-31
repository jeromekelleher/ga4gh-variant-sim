#!/bin/bash

for k in `seq 2 7`; do
	echo Simulate 10**$k
	python3 ../simulator.py generate-trees -L 10 -p $k k"$k"_L10.trees
done
