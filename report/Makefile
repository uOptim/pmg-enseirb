OBJECTS=report.pdf

all: ${OBJECTS}

%.pdf: %.tex
	pdflatex $< && pdflatex $<


clean:
	rm -f *.log *.aux *.out *.toc

mrproper: clean
	rm -f *.pdf