# Changelog

## 0.5.1 (Dec 2022)
* Now support Python3.10
* Bugfix #67: When the channels argument of Aserver and Arecorder has not been set it was determined by the default device instead of the actual device.

## 0.5.0 (Oct 2022)
* Make Aserver a context that can wrap play inside. This context will handle boot() and quit() automatically.
* Move OSX/Linux CI to Github Action from TravisCI.
* Bump numpy dependency, numpy.linspace usage is updated.
* Make Asig.play() arguments explicit. 
* Asig.dur property that returns the duration in seconds of the signal

## 0.4.3 (March 2021)
* Add logging helper function.
* Make sure `pya` runs on Python3.8
* Bugfix #43: Handle `None` as a return value of `get_server_info` in `pya.helper.backend.determine_backend` (thanks @paulvickers)

## 0.4.1 (July 2020)
* Fix bug of extend mode result in a change of dtype (from the default f32 to f64). Now it is fixed at f32.
* Add convolve() method. A convolution method (default to fft). Now you can apply reverb, do autocorrelation to the signal.
* Update Ugen docstring and remove deprecated variable names
* namespace improvement.
* `sanic` and `notebook` are now optional: If you plan to use `JupyterBackend`, install `pya` with `pip install pya[remote]`

## 0.4.0 (April 2020)
* new Amfcc class for MFCC feature extraction
* new helper.visualization.py and gridplot()
* optimise plot() methods of all classes for better consistency in both style and the API
* new helper functions: padding, is_pow2, next_pow2, round_half_up, rolling_window, signal_to_frame,
  magspec, powspec, hz2mel, mel2hz_
* Fixed bug that for multichanels spectrum the rfft is not performed on the x-axis

## 0.3.3 (February 2020)

* Bugfix #34: Improve indexing of Asig
* Added developer tools for documentation generation
* Improved string handling to prevent Python 3.8 warnings
* Fixed reverse proxy resolution in JupyterBackend for Binder hosts other than Binderhub
* Fixed travis job configuration

## 0.3.2 (November 2019)

* Fixed bug of multi channel fade_in fade_out
* Introduced timeslice in extend setitem mode as long as stop reaches the end of the array.
* Added Sphinx documentation which can be found at https://interactive-sonification.github.io/pya
* Introduced new Arecorder method to individualize channel selection and gain adjustment to each channel
* Added Binder links to Readme
* Introduced Jupyter backend based on WebAudio
* Updated example notebook to use WebAudio when used with Binder
* Switched test framework from nosetests to pytest


## 0.3.1 (October 2019)

* Remove ffmpeg from requirement, and it has to be installed via conda or manually
* Decouple pyaudio from Aserver and Arecorder
* Introduce backends interface: PyAudio(), Dummy()
* add device_info() helper function which prints audio device information in a tabular form
* Bugfix #23: Add a small delay to server booting to prevent issues when Aserver and Arecorder are initialized back-to-back
* Helper function `record` has been dropped due to legacy reasons. An improved version will be introduced in 0.3.2.


## 0.3.0 (October 2019)

* Restructure Asig, Astft, Aspec into different files
* Add Arecorder class
* Several bug fixes
* Made ffmpeg optional
* Introduced CI testing with travis (*nix) and appveyor (Windows)
* Introduced coveralls test coverage analysis


## 0.2.1 (August 2019)

* 0.2.1 is a minor update that removes the audioread dependency and directly opt for standard library for .wav and .aif support, ffmpeg for .mp3 support.
* Bugfix for multichannels audio file loading resulting transposed columns and rows.


## 0.2 (August 2019)

* First official PyPI release

