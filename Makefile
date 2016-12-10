-include configuration.mk

NOWEAVE ?= noweave
NOTANGLE ?= notangle
TEX2PDF ?= lualatex
BIBTEX ?= bibtex
CPIF ?= cpif
PYTHON ?= python3
F2PY ?= f2py
AUTOPEP8 ?= autopep8
DOCKER ?= sudo docker
MPIRUN ?= mpirun -n 2
INKSCAPE ?= inkscape
MV ?= mv

DEPS = \
	figures/TOD_indexing.pdf \
	figures/core_scan_example.pdf \
	figures/baseline_calculation.pdf \
	figures/period_lengths.pdf \
	figures/1fnoise.pdf \
	figures/FITS_structure.pdf \
	figures/operators.pdf \
	figures/split_MPI_1.pdf \
	figures/split_MPI_2.pdf \
        figures/split_TOD_files_1.pdf \
        figures/split_TOD_files_2.pdf \
	figures/OfsAndGains_structure_1.pdf \
	figures/OfsAndGains_structure_2.pdf \
	figures/diagm.pdf \
	figures/test_dipole_temperature.pdf \
	figures/long_test_scanning.pdf \
        figures/long_test_hits.pdf \
	test_split_into_periods.txt \
	test_split_into_n.txt \
	test_split.txt \
	test_assign_files_to_processes.txt \
	test_sum_subranges.txt

.phony: all docker check fullcheck help long_test_files

all: calibrate.py index.py dacapo_calibration.pdf dacapo-test.py

check: all
	$(PYTHON) -m 'unittest' dacapo-test.py

fullcheck: check index.py calibrate.py check-gains.py long_test_files
	$(PYTHON) ./index.py test_files/long_test_index.ini && \
	$(MPIRUN) $(PYTHON) ./calibrate.py test_files/long_test_calibrate_none.ini && \
	$(PYTHON) ./check-gains.py test_files/long_test_tod.fits test_files/long_test_results_none.fits && \
	$(MPIRUN) $$(PYTHON) ./calibrate.py test_files/long_test_calibrate_jacobi.ini && \
	$(PYTHON) ./check-gains.py test_files/long_test_tod.fits test_files/long_test_results_jacobi.fits && \
	$(MPIRUN) $$(PYTHON) ./calibrate.py test_files/long_test_calibrate_full.ini && \
	$(PYTHON) ./check-gains.py test_files/long_test_tod.fits test_files/long_test_results_full.fits

long_test_files: create-test-files.py test_files/long_test_tod.fits test_files/long_test_hits.fits.gz
	$(PYTHON) ./create-test-files.py test_files

help:
	@echo "Usage: make COMMAND"
	@echo ""
	@echo "where COMMAND can be one of:"
	@echo ""
	@echo "    all         Build all the binary files and the PDF documents"
	@echo "    docker      Run Docker and check that the build works"
	@echo "    check       Run a set of unit tests on the code (quick)"
	@echo "    fullcheck   Run all the unit tests and the integration tests (slow)"

docker: docker/Dockerfile
	cd docker && $(DOCKER) build -t="ziotom78:dacapo" ./
	@echo "Run docker with the command"
	@echo "    docker run --name dacapo \"ziotom78:dacapo\""

dacapo_calibration.pdf: dacapo_calibration.tex dacapo_calibration.bbl $(DEPS)
	$(TEX2PDF) $< && $(TEX2PDF) $<

dacapo_calibration.bbl: dacapo_calibration.bib
	$(BIBTEX) dacapo_calibration

docker/Dockerfile: dacapo_calibration.nw
	$(NOTANGLE) -RDockerfile $^ | $(CPIF) $@

figures/core_scan_example.pdf: figures/core_scan_example.fits figures/core_scan_example.py
	$(PYTHON) figures/core_scan_example.py figures/core_scan_example.fits $@

# For this recipe, *first* build the .so library, then build calibrate.py. In this
# way, if f2py fails, calibrate.py is not generated and the rule will be reapplied
# the next time "make" is called
calibrate.py: dacapo_calibration.nw
	$(NOTANGLE) -Rftnroutines.f90 $^ | $(CPIF) ftnroutines.f90 && \
		$(F2PY) -c -m ftnroutines --f90flags=-std=f2003 ftnroutines.f90
	$(NOTANGLE) -R$@ $^ | $(CPIF) $@
	$(AUTOPEP8) --in-place $@

index.py: dacapo_calibration.nw
	$(NOTANGLE) -R$@ $^ | $(CPIF) $@
	$(AUTOPEP8) --in-place $@

%.tex: %.nw
	$(NOWEAVE) -n -delay -index $< | $(CPIF) $@

./%.py: dacapo_calibration.nw
	$(NOTANGLE) -R$@ $^ | $(CPIF) $@
	$(AUTOPEP8) --in-place $@

%.pdf: %.svg
	$(INKSCAPE) --export-pdf=$@ $<

test_split_into_periods.txt: index.py scripts/test_split_into_periods.py
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) scripts/test_split_into_periods.py | $(CPIF) $@

test_split_into_n.txt: calibrate.py scripts/test_split_into_n.py
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) scripts/test_split_into_n.py | $(CPIF) $@

test_split.txt: calibrate.py scripts/test_split.py
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) scripts/test_split.py | $(CPIF) $@

test_assign_files_to_processes.txt: calibrate.py scripts/test_assign_files_to_processes.py
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) scripts/test_assign_files_to_processes.py | $(CPIF) $@

test_sum_subranges.txt: calibrate.py scripts/test_sum_subranges.py
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) scripts/test_sum_subranges.py | $(CPIF) $@

figures/test_dipole_temperature.pdf: calibrate.py  scripts/test_dipole_temperature.py
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) scripts/test_dipole_temperature.py && \
		$(MV) test_dipole_temperature.pdf figures/

figures/1fnoise.pdf: figures/noise.dat figures/noise_plot.py
	PYTHONPATH=.:$(PYTHONPATH) $(PYTHON) figures/noise_plot.py figures/noise.dat $@
