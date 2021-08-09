#!/bin/bash

for k in `seq 2 7`; do
	echo Simulate 10**$k
    STEM=k"$k"_L20
	# python3 ../simulator.py generate-trees -L 20 -p $k $STEM.trees
    # make -j 2 $STEM.vcf.gz $STEM.sav $STEM.sgz
    make -j 2 "$STEM"_GQ.vcf.gz "$STEM"_GQ.sav "$STEM"_GQ.sgz
done
