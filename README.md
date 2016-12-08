# Calibrating TODs using the DaCapo algorithm

This repository contains the implementation of the calibration
algorithm used in the
[Planck](http://www.cosmos.esa.int/web/planck)/LFI
[2015 data release](http://www.cosmos.esa.int/web/planck/pla). It is a
Python3/Fortran program developed using
[literate programming techniques](https://en.wikipedia.org/wiki/Literate_programming),
and it uses MPI to distribute the workload among several processing
units.

The full source code of the program can be found in the PDF document
[dacapo_calibration.pdf](https://github.com/ziotom78/dacapo_calibration/blob/master/dacapo_calibration.pdf).

## Installation

After you have downloaded the repository, just type `make`. You can
configure the programs `make` calls by setting variables in a new
file named `configuration.mk`. The following variables are
accepted:
- `NOWEAVE`
- `NOTANGLE`
- `CPIF`
- `TEX2PDF`
- `PYTHON`
- `AUTOPEP8`
- `DOCKER`
- `MPIRUN`
- `INKSCAPE`

## License

The code is released under a permissive MIT license. See the file
[LICENSE](https://github.com/ziotom78/dacapo_calibration/blob/master/LICENSE).

## Citation

If you use this code in your publications, please cite this GitHub
repository. (Things will change once a paper about this work will be
published. Stay tuned!)
