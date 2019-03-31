# """Collection of classes and functions for processing audio signals 
# in python and jupyter notebooks, for synthesis, effects, analysis and plotting.
# """
from .pya import Asig, Aspec, Astft
from .helpers import ampdb, dbamp, cpsmidi, midicps, linlin, clip, record, playpyaudio
from .pyaudiostream import *