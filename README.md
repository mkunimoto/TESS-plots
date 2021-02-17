# TESS-plots

This package includes two tools useful for assessing the candidacy of planet candidates found in TESS data.

## 1. Difference images

A common source of false positives in transit surveys is signals that originate from stars nearby or in the background of the target. Light from these contaminant stars contributes to the lightcurve, so a deep eclipse around a nearby star can appear like a planet transit around the target. Since TESS pixels are large (21"x21"), this is especially important to consider when assessing the candidacy of TESS planet candidates.

This tool takes in basic information about the target star (TIC ID, RA, Dec) and the potential planet (period, epoch, duration), uses TESScut to create a cutout of the FFIs centred on the target, identifies in- and out-of-transit cadences, and computes and plots difference images. See `example/example_diffimages.ipynb` for an example use of this tool.

Requirements: 
- astropy
- astroquery
- matplotlib
- numpy
- pandas
- pickle
- tess-point

This tool uses TESS Spice kernels to compute the barycentric time correction for the specific target.  In addition to the included TESS merged ephemeris file (.bsp) and leap seconds file (.tls), this requires a planetary ephemeris file, which is larger than the maximum file size allowed by GitHub.  Before you can use this tool you must download the planeary ephemeris file, found at https://archive.stsci.edu/missions/tess/models/tess2018338154429-41241_de430.bsp

## 2. Diagnostic plots

This tools plots TESS lightcurves in various views useful for visually reviewing planet candidates. In particular, raw and detrended lightcurves, phase diagrams, odd- and even- transits, and potential secondaries in the lightcurve are shown, which are all informative ways to identify false positives and false alarms. The plot also displays the differences images from above on the same page, so both flux- and pixel-level vetting are available. See `example/example_diagnosticplot.ipynb` for an example use of this tool.

Requirements:
- astropy
- astroquery
- brokenaxes
- matplotlib
- numpy
- scipy

## TODO:
- some sections of code assume a 20x20 pixel cutout; make this more general 
