# Changelog

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

