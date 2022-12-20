[![PyPI](https://img.shields.io/pypi/v/pya.svg)](https://pypi.org/project/pya)
[![License](https://img.shields.io/github/license/interactive-sonification/pya.svg)](LICENSE)

# pya

|Branch|`master`|`develop`|
|------:|--------:|---------:|
|[CI-Linux/MacOS](https://github.com/interactive-sonification/pya/actions/workflows/pya-ci.yaml) | [![Build Status Master](https://github.com/interactive-sonification/pya/actions/workflows/pya-ci.yaml/badge.svg?branch=master)](https://github.com/interactive-sonification/pya/actions/workflows/pya-ci.yaml?query=branch%3Amaster) | [![Build Status Develop](https://github.com/interactive-sonification/pya/actions/workflows/pya-ci.yaml/badge.svg?branch=develop)](https://github.com/interactive-sonification/pya/actions/workflows/pya-ci.yaml?query=branch%3Adevelop) |
|[CI-Windows](https://ci.appveyor.com/project/aleneum/pya-b7gkx/)| ![Build status AppVeyor](https://ci.appveyor.com/api/projects/status/vn61qeri0uyxeedv/branch/master?svg=true) | ![Build status AppVeyor](https://ci.appveyor.com/api/projects/status/vn61qeri0uyxeedv/branch/develop?svg=true) | 
|Changes|[![GitHub commits](https://img.shields.io/github/commits-since/interactive-sonification/pya/v0.5.0/master.svg)](https://github.com/interactive-sonification/pya/compare/v0.5.0...master) | [![GitHub commits](https://img.shields.io/github/commits-since/interactive-sonification/pya/v0.5.0/develop.svg)](https://github.com/interactive-sonification/pya/compare/v0.5.0...develop) |
|Binder|[![Master Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/interactive-sonification/pya/master?filepath=examples%2Fpya-examples.ipynb) | [![Develop Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/interactive-sonification/pya/develop?filepath=examples%2Fpya-examples.ipynb) |

## What is pya?

pya is a package to support creation and manipulation of audio signals with Python.
It uses numpy arrays to store and compute audio signals.

  * Documentation: see examples/pya-examples.ipynb for a quick tutorial and [Documentation](https://interactive-sonification.github.io/pya/index.html)
  * Source code: https://github.com/interactive-sonification/pya

It provides:

  * Asig - a versatile audio signal class 
      * Ugen - a subclass of Asig, which offers unit generators 
        such as sine, square, sawtooth, noise
  * Aserver - an audio server class for queuing and playing Asigs
  * Arecorder - an audio recorder class
  * Aspec - an audio spectrum class, using rfft as real-valued signals are always implied
  * Astft - an audio STFT (short-term Fourier transform) class
  * A number of helper functions, e.g. device_info()

pya can be used for
* multi-channel audio processing
* auditory display and sonification
* sound synthesis experiment
* audio applications in general such as games or GUI-enhancements
* signal analysis and plotting
  
At this time pya is more suitable for offline rendering than realtime.

## Authors and Contributors

* [Thomas](https://github.com/thomas-hermann) (author, maintainer)
* [Jiajun](https://github.com/wiccy46) (co-author, maintainer)
* [Alexander](https://github.com/aleneum) (maintainer)
* Contributors will be acknowledged here, contributions are welcome.

## Installation

`pya` requires `portaudio` and its Python wrapper `PyAudio` to play and record audio. 

### Using Conda

Pyaudio can be installed via [conda](https://docs.conda.io):

```
conda install pyaudio
```

Disclaimer: Python 3.10+ requires PyAudio 0.2.12 which is not available on Conda as of December 2022. [Conda-forge](https://conda-forge.org/) provides a version only for Linux at the moment. Users of Python 3.10 should for now use other installation options.

### Using Homebrew and PIP (MacOS only)


```
brew install portaudio
```

Then  

```
pip install pya
```

For Apple ARM Chip, if you failed to install the PyAudio dependency, you can follow this guide: [Installation on ARM chip](https://stackoverflow.com/a/73166852/4930109)
  - Option 1: Create .pydistutils.cfg in your home directory, `~/.pydistutils.cfg`, add:

    ```
    echo "[build_ext]
    include_dirs=$(brew --prefix portaudio)/include/
    library_dirs=$(brew --prefix portaudio)/lib/" > ~/.pydistutils.cfg
    ```
    Use pip:

    ```
    pip install pya
    ```

    You can remove the `.pydistutils.cfg` file after installation.

- Option 2: Use `CFLAGS`: 

    ```
    CFLAGS="-I/opt/homebrew/include -L/opt/homebrew/lib" pip install pya
    ```



### Using PIP (Linux)

Try `sudo apt-get install portaudio19-dev` or equivalent to your distro, then 

```
pip isntall pya
```

### Using PIP (Windows)

[PyPI](https://pypi.org/) provides [PyAudio wheels](https://pypi.org/project/PyAudio/#files) for Windows including portaudio:

```
pip install pyaudio
```

should be sufficient.


## A simple example

### Startup:

```Python
import pya
s = pya.Aserver(bs=1024)
pya.Aserver.default = s  # to set as default server
s.boot()
```

### Create an Asig signal:

A 1s / 440 Hz sine tone at sampling rate 44100 as channel name 'left':

```Python
import numpy as np
signal_array = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
atone = pya.Asig(signal_array, sr=44100, label='1s sine tone', cn=['left'])
```

Other ways of creating an Asig object:

```Python
asig_int = pya.Asig(44100, sr=44100)  # zero array with 44100 samples
asig_float = pya.Asig(2., sr=44100)  # float argument, 2 seconds of zero array
asig_str = pya.Asig('./song.wav')  # load audio file
asig_ugen = pya.Ugen().square(freq=440, sr=44100, dur=2., amp=0.5)  # using Ugen class to create common waveforms
```

Audio files are also possible using the file path. `WAV` should work without issues. `MP3` is supported but may raise error if [FFmpeg](https://ffmpeg.org/).

If you use Anaconda, installation is quite easy:

`conda install -c conda-forge ffmpeg`

Otherwise:

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

Instead of specifying a long standing server. You can also use `Aserver` as a context:

```Python
with pya.Aserver(sr=48000, bs=256, channels=2) as aserver:
    atone.play(server=aserver)  # Or do: aserver.play(atone)
```

The benefit of this is that it will handle server bootup and shutdown for you. But notice that server up/down introduces extra latency.

### Play signal on a specific device

```Python
from pya import find_device
from pya import Aserver
devices = find_device() # This will return a dictionary of all devices, with their index, name, channels.
s = Aserver(sr=48000, bs=256, device=devices['name_of_your_device']['index'])
```


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

### Recording from Device

`Arecorder` allows recording from input device 

```Python
import time

from pya import find_device
from pya import Arecorder
devices = find_device()  # Find the index of the input device
arecorder = Arecorder(device=some_index, sr=48000, bs=512)  # Or not set device to let pya find the default device 
arecorder.boot()
arecorder.record()
time.sleep(2)  # Recording is non-blocking
arecorder.stop()
last_recording = arecorder.recordings[-1]  # Each time a recorder stop, a new recording is appended to recordings
```

### Method chaining
Asig methods usually return an Asig, so methods can be chained, e.g

    atone[{0:1.5}].fade_in(0.1).fade_out(0.8).gain(db=-6).plot(lw=0.1).play(rate=0.4, onset=1)

### Learning more
* Please check the examples/pya-examples.ipynb for more examples and details.


## Contributing 
* Please get in touch with us if you wish to contribute. We are happy to be involved in the discussion of new features and to receive pull requests.

