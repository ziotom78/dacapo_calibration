# Version 1.1.1

- It is now possible to specify an empty `mask =` in the `dacapo`
  section of the configuration file. This has the same effect as
  not using `mask` at all, and it should ease the creation of
  configuration files through text templates
- Remove "clobber"-related warnings, at the expense of requiring AstroPy 1.3 at
  least

# Version 1.1

- Relativistic quadrupolar correction (`frequency_hz` in the configuration file
  used by `calibrate.py`)
- Possibility to use a subset of the TODs listed in a FITS file (`first_index`
  and `last_index` in the configuration file used by `calibrate.py`)

# Version 1.0

- Initial release
- Used to produce the CORE papers
