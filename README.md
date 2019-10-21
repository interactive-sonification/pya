[![Version](https://img.shields.io/badge/version-v0.3.1-orange.svg)](https://github.com/interactive-sonification/pya)
[![PyPI](https://img.shields.io/pypi/v/pya.svg)](https://pypi.org/project/pya)
[![GitHub commits](https://img.shields.io/github/commits-since/interactive-sonification/pya/v0.3.0.svg)](https://github.com/interactive-sonification/pya/compare/v0.3.0...master)
[![License](https://img.shields.io/github/license/interactive-sonification/pya.svg)](LICENSE)
[![Build Status Travis](https://travis-ci.org/interactive-sonification/pya.svg?branch=develop)](https://travis-ci.org/interactive-sonification/pya)
[![Build status AppVeyor](https://ci.appveyor.com/api/projects/status/vn61qeri0uyxeedv/branch/develop?svg=true)](https://ci.appveyor.com/project/aleneum/pya-b7gkx/branch/develop)
<!--
[![Coverage Status](https://coveralls.io/repos/interactive-sonification/pya/badge.svg?branch=master&service=github)](https://coveralls.io/github/interactive-sonification/pya?branch=master)
-->

<!--[![Name](Image)](Link)-->

# pyA

## What is pyA?

pyA is a package to support creation and manipulation of audio signals with Python.
It uses numpy arrays to store and compute audio signals.

  * Website: --
  * Documentation: see examples/pya-examples.ipynb and help(pya)
  * Source code: https://github.com/interactive-sonification/pya

It provides:

  * Asig - a versatile audio signal class 
      * Ugen - a subclass of Asig, which offers unit generators 
        such as sine, square, swatooth, noise
  * Aserver - an audio server class for queuing and playing Asigs
  * Arecorder - an audio recorder class
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
* Alexander Neumann, Neurocognitions and Action - Biomechanics, Bielefeld University
* Contributors will be acknowledged here, contributions are welcome.

## Installation

<!-- **Disclaimer**: We are currently making sure that pyA can be uploaded to PyPI, until then clone the master branch and from inside the pya directory install via `pip install -e .` -->

**Note**: pya can be installed using **pip**. But pya uses PyAudio for audio playback and record, and PyAudio 0.2.11 has yet to fully support Python 3.7. So using pip install with Python 3.7 may encounter issues such as portaudio. Solution

Use pip to install pya via

    pip install pya

**Note**: pyA requires PyAudio, for playback and record, which in turn requires portaudio. Since PyAudio 0.2.11 has yet to fully support Python 3.7, you maybe encounter portaudio related error under Python 3.7. Lower versions of Python should be fine. A few solutions are:


1. Anaconda can install non-python packages, so that the easiest way (if applicable) would be to 

    conda install pyaudio

2. For Mac users, you can `brew install portaudio` beforehand. 

3. For Linux users, try `sudo apt-get install portaudio19-dev` or equivalent to your distro.

4. For Windows users, you can install PyAudio wheel at:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio




See pyaudio installation http://people.csail.mit.edu/hubert/pyaudio/#downloads

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

Other ways of creating an Asig object:

    asig_int = Asig(44100, sr=44100)  # zero array with 44100 samples
    asig_float = Asig(2., sr=44100)  # float argument, 2 seconds of zero array
    asig_str = Asig('./song.wav')  # load audio file
    asig_ugen = Ugen().square(freq=440, sr=44100, dur=2., amp=0.5)  # using Ugen class to create common waveforms

Audio files are also possible using the file path. `WAV` should work without issues. `MP3` is supported but may raise error if [FFmpeg](https://ffmpeg.org/).

* Mac or Linux with brew
    - `brew install ffmpeg`
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

