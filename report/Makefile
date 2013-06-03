OBJECTS=report.pdf

all: ${OBJECTS}

figures/collision.png: comparaison_collision.gnuplot
	gnuplot $<

%.pdf: %.tex clean figures/collision.png
	pdflatex $< && pdflatex $<


clean:
	rm -f *.log *.aux *.out *.toc *.lof

mrproper: clean
	rm -f *.pdf