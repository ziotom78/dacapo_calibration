from index import TODFileInfo
from calibrate import assign_files_to_processes
files = [TODFileInfo(name, 0, 12, 12) for name in ('A.fits',
                                                   'B.fits',
                                                   'C.fits')]
result = assign_files_to_processes([10, 10, 8, 8], files)
for mpi_idx, proc in enumerate(result):
    for subrange in proc:
        print('Process #{0}: {1}, {2:2d} |{3}|'
              .format(mpi_idx + 1,
                      subrange.file_info.file_name,
                      subrange.first_idx,
                      '-' * subrange.num_of_samples))
