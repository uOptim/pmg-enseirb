OBJECTS=report.pdf

all: ${OBJECTS}

figures: figures/collision_slices.png   \
         figures/collision_versions.png \
         figures/lennard_slices.png     \
         figures/lennard_versions.png

figures/collision_versions.png: comparaison_collision.gnuplot
	gnuplot $<

figures/collision_slices.png: comparaison_slice_collision.gnuplot
	gnuplot $<


figures/lennard_versions.png: comparaison_lennard.gnuplot
	gnuplot $<

figures/lennard_slices.png: comparaison_slice_lennard.gnuplot
	gnuplot $<

%.pdf: %.tex clean figures
	pdflatex $< && pdflatex $<


clean:
	rm -f *.log *.aux *.out *.toc *.lof

mrproper: clean
	rm -f *.pdf