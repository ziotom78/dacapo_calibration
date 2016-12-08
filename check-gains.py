#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''Usage: {basename} FILE1 FILE2

Check the consistency between the gains in FILE1 and FILE2. The two
files can have been created either using create-test-files.py or
calibrate.py.
'''

import os.path
import sys
from astropy.io import fits
import numpy as np


def main():
    if len(sys.argv) != 3:
        print(__doc__.format(basename=os.path.basename(sys.argv[0])))
        sys.exit(1)

    file1_name, file2_name = sys.argv[1:3]

    with fits.open(file1_name) as f:
        file1_gains = f['GAINS'].data.field('GAIN')

    with fits.open(file2_name) as f:
        file2_gains = f['GAINS'].data.field('GAIN')

    assert np.allclose(file1_gains, file2_gains,
                       rtol=2e-2), 'Gains do not match'


if __name__ == '__main__':
    main()
