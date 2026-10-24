# MRI sensitivity maps

This package standardizes the supplied maps, aligns them to reconstruction space,
and validates their SENSE convention for the 148 paired volumes.

- Supplied `[slice, height, width, coil]` maps become
  `[slice, coil, height, width]` on load.
- The 18 supplied 320x320 mismatches are center-cropped to the manifest
  reconstruction shape.
- The convention is `coil_image = sensitivity * object`; the SENSE adjoint uses
  the complex conjugate of the map.
Verification covered one representative slice from every supplied-map volume: all
148 passed, including the 18 cropped mismatches. Median SENSE-vs-RSS NRMSE inside
map support was 0.059. No sensitivity maps are estimated by this project.
