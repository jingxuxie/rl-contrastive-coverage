PYTHON ?= python
BIBTEX ?= $(shell command -v bibtex 2>/dev/null || command -v bibtex.original)
PYTHONPATH := src
TEXINPUTS := ../AAAI_AuthorKit27//:
BSTINPUTS := ../AAAI_AuthorKit27//:

.PHONY: test experiments paper supplement preflight submission-bundle clean all

all: test experiments paper supplement

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m compileall -q src experiments tests

experiments:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/run_all.py

paper: experiments
	cd paper && TEXINPUTS=$(TEXINPUTS) pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && BSTINPUTS=$(BSTINPUTS) $(BIBTEX) main
	cd paper && TEXINPUTS=$(TEXINPUTS) pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && TEXINPUTS=$(TEXINPUTS) pdflatex -interaction=nonstopmode -halt-on-error main.tex

supplement:
	cd paper && TEXINPUTS=$(TEXINPUTS) pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
	cd paper && BSTINPUTS=$(BSTINPUTS) $(BIBTEX) supplement
	cd paper && TEXINPUTS=$(TEXINPUTS) pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
	cd paper && TEXINPUTS=$(TEXINPUTS) pdflatex -interaction=nonstopmode -halt-on-error supplement.tex

preflight: all
	$(PYTHON) scripts/preflight_submission.py

submission-bundle: preflight
	$(PYTHON) scripts/package_submission.py

clean:
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.toc
