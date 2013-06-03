OBJECTS=report.pdf

all: ${OBJECTS}

%.pdf: %.tex clean
	pdflatex $< && pdflatex $<


clean:
	rm -f *.log *.aux *.out *.toc

mrproper: clean
	rm -f *.pdf