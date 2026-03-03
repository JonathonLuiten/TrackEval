# 1.4.0

 - Remove --METRICS argument from the KITTI CLI
 - Start adding typehints in the submodule `trackeval.eval`.
 - Add `tqdm` as dependency.
 - End of support for Python 3.8.

# 1.3.0

 - Python 3.14 support
 - Update dependencies
 - Change to hatchling build system
 - Default output directory is empty string rather than `None`.
 - Added some typing hints, primarily in HOTA.
 - Specifying the output folder in unit test so the test src not polluted.
 - Testing only pedestrian class to speed up.
 - There is no need to insist on presence of files in tests folder since repo is open.

