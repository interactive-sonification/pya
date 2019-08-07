[![Version](https://img.shields.io/badge/version-v0.2-orange.svg)](https://github.com/Thomas-Hermann/pya)
[![PyPI](https://img.shields.io/pypi/v/pya.svg)](https://pypi.org/project/pya)
[![License](https://img.shields.io/github/license/Thomas-Hermann/pya.svg)](LICENSE)
<!-- 
[![Build Status](https://travis-ci.org/Thomas-Hermann/pya.svg?branch=master)](https://travis-ci.org/Thomas-Hermann/pya)
[![Coverage Status](https://coveralls.io/repos/Thomas-Hermann/pya/badge.svg?branch=master&service=github)](https://coveralls.io/github/Thomas-Hermann/pya?branch=master)

[![GitHub commits](https://img.shields.io/github/commits-since/Thomas-Hermann/pya/0.2.svg)](https://github.com/Thomas-Hermann/pya/compare/0.2...master) -->
<!--[![Name](Image)](Link)-->

# pyA

## What is pyA?

pyA is a package to support creation and manipulation of audio signals with Python.
It uses numpy arrays to store and compute audio signals.

  * Website: --
  * Documentation: see examples/pya-examples.ipynb and help(pya)
  * Source code: https://github.com/thomas-hermann/pya

It provides:

  * Asig - a versatile audio signal class 
      * Ugen - a subclass of Asig, which offers unit generators 
        such as sine, square, swatooth, noise
  * Aserver - an audio server class for queuing and playing Asigs
  * Aspec - an audio spectrum class, using rfft as real-valued signals are always implied
  * Astft - an audio STFT (short-term Fourier transform) class

pyA can be used for
* multi-channel audio processing
* auditory display and sonification
* sound synthesis experiment
* audio applications in general such as games or GUI-enhancements
* signal analysis and plotting
* at this time more suitable for offline rendering than realtime.

## Authors and Contributors

* Thomas Hermann, Ambient Intelligence Group, Faculty of Technology, Bielefeld University (author and maintainer)
* Jiajun Yang, Ambient Intelligence Group, Faculty of Technology, Bielefeld University (co-author)
* Contributors will be acknowledged here, contributions are welcome.

## Installation

**Disclaimer**: We are currently making sure that pyA can be uploaded to PyPI, until then clone the master branch and from inside the pya directory install via `pip install -e .`

Use pip to install pya via

    pip install pya

Besides numpy, scipy and matplotlib, pyA requires pyaudio which in turn requires portaudio. Note that Linux/Windows users should make sure that the corresponding binary packages are installed.

See pyaudio installation http://people.csail.mit.edu/hubert/pyaudio/#downloads

Anaconda can install non-python packages, so that the easiest way (if applicable) would be to 

    conda install pyaudio

## A simple example

### Startup:

    from pya import *
    s = Aserver(bs=1024)
    Aserver.default = s  # to set as default server
    s.boot()   

### Create an Asig signal:

A 1s / 440 Hz sine tone at sampling rate 44100 as channel name 'left':

    import numpy as np
    signal_array = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    atone = Asig(signal_array, sr=44100, label='1s sine tone', cn=['left'])

In addition to passing numpy.ndarray as argument for Asig, it is also for possible to pass int as samples or float as seconds to create an Asig obj with silence audio. 

Audio files are also possible using the file path. `WAV` should work without issues. `MP3` is supported but may raise error under Windows and Linux if [FFmpeg](https://ffmpeg.org/) is not installed. MacOS should be fine thanks to CoreAudio.

* On Linux
    - Install FFmpeg via apt-get: `sudo apt install ffmpeg`
* On Windows
    - Download the latest distribution from https://ffmpeg.zeranoe.com/builds/
    - Unzip the folder, preferably to `C:\`
    - Append the FFmpeg binary folder (e.g. `C:\ffmpeg\bin`) to the PATH system variable ([How do I set or change the PATH system variable?](https://www.java.com/en/download/help/path.xml))
### Key attributes
* `atone.sig`  --> The numpy array containing the signal is 
* `atone.sr`  --> the sampling rate
* `atone.cn` --> the list of custom defined channel names
* `atone.label` --> a custom set identifier string

### Play signals

    atone.play(server=s)  

play() uses Aserver.default if server is not specified

### Plotting signals

to plot the first 1000 samples:

    atone[:1000].plot()

to plot the magnitude and phase spectrum:

    atone.plot_spectrum()

to plot the spectrum via the Aspec class

   atone.to_spec().plot()

to plot the spectrogram via the Astft class

    atone.to_stft().plot(ampdb)

### Selection of subsets
* Asigs support multi-channel audio (as columns of the signal array)
  * `a1[:100, :3]` would select the first 100 samples and the first 3 channels, 
  * `a1[{1.2:2}, ['left']]` would select the channel named 'left' using a time slice from 1

### Method chaining
Asig methods usually return an Asig, so methods can be chained, e.g

    atone[{0:1.5}].fade_in(0.1).fade_out(0.8).gain(db=-6).plot(lw=0.1).play(rate=0.4, onset=1)

### Learning more
* Please check the examples/pya-examples.ipynb for more examples and details.


## Contributing 
* Please get in touch with us if you wish to contribute. We are happy to be involved in the discussion of new features and to receive pull requests.

