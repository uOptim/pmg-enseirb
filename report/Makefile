OBJECTS=report.pdf

all: ${OBJECTS}

%.pdf: %.tex
	pdflatex $< && pdflatex $<


clean:
	rm *.log
	rm *.aux
	rm *.out
	rm *.toc

mrproper: clean
	rm *.pdf