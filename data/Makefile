# Makefile to automate converting datasets.
#

%.vcf.gz: %.trees
	python3 ../simulator.py trees-to-vcf $< - | gzip > $@

%_GQ.vcf.gz: %.trees
	python3 ../simulator.py trees-to-vcf --simulate-gq $< - | gzip > $@

%.sgz: %.trees
	python3 ../simulator.py trees-to-sgkit-zarr --variant-chunk-size=1000 $< $@

%_GQ.sgz: %.trees
	python3 ../simulator.py trees-to-sgkit-zarr --variant-chunk-size=1000 --simulate-gq $< $@

%.sav: %.vcf.gz
	./sav import --phasing full $< $@
	
