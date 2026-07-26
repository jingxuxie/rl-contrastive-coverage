PYTHON ?= python
BIBTEX ?= $(shell command -v bibtex 2>/dev/null || command -v bibtex.original)
PYTHONPATH := src

.PHONY: test experiments paper supplement clean all

all: test experiments paper supplement

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m compileall -q src experiments tests

experiments:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/run_all.py

paper: experiments
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && $(BIBTEX) main
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex

supplement:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
	cd paper && $(BIBTEX) supplement
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error supplement.tex

clean:
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.toc
