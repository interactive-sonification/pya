AUDIO_RATE: int = 44100
"""Audio rate in Hz."""

CONTROL_RATE: int = 100
"""Control rate in Hz."""

MAX_GEN_TIME: int | float | None = 15 * 60
"""Maximum length of generated asigs in seconds. Default is 15 minutes.

This is to prevent the memory filling up when accidentally calling gen_asig() without
any arguments on an Asig that will not terminate.

To disable this limit, set MAX_GEN_TIME to None.
"""

UPSAMPLE_WARNING: bool = True
"""Whether to print a warning when upsampling an Asig as this is currently experimental
and does not always produce expected results.
"""

PLOT_SEQUENCE_AS_GRAPH_WARNING: bool = True
"""Whether to print a warning when plotting a sequence as a graph if the sequence is too
long. This is to prevent the graph from being too large and unreadable.
"""
