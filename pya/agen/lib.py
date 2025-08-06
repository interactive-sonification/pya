from __future__ import annotations

import math
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
from pya.asig import Asig
from scipy.signal import lfilter, lfiltic

from pya.agen import config
from pya.agen.core import (
    AddGen,
    AGen,
    MultiChannelGen,
    SingleChannelGen,
    get_max_channel,
    get_max_sr,
)

if TYPE_CHECKING:
    from pya.agen.types import GenOrNum

# Numba fallback
try:
    from numba import float64, njit, int64
    from numba.types import Tuple
except ImportError:
    warnings.warn("Numba is not installed. Some generators may be slower.")
    def njit(*args, **kwargs):
        def decorator(func):
            def wrapper(*args, **kwargs):
                warnings.warn(
                    "Using raw Python fallback. Please install numba for better performance.", 
                )
                return func(*args, **kwargs)
            return wrapper
        return decorator

    class DummyType:
        def __getitem__(self, item):
            return self
        def __call__(self, *args, **kwargs):
            return self

    Tuple, float64, int64 = DummyType(), DummyType(), DummyType()



class ChannelMappingStrategy(str, Enum):
    CYCLE = "cycle"
    """Repeat the channels of the generator in a cycle."""
    LAST = "last"
    """Repeat the last channel of the generator."""
    ZERO = "zero"
    """Set the additional channels to zero."""


def expand_channels(
    gen: AGen,
    target_count: int,
    strategy: ChannelMappingStrategy,
) -> MultiChannelGen:
    """Expand the channels of a generator to a target count using a mapping strategy.

    Parameters
    ----------
    gen
        The generator to expand.
    target_count
        The target number of channels.
    strategy
        The mapping strategy to use.

    Returns
    -------
    MultiChannelGen
        The generator with the expanded channels.

    """
    if gen.channels > target_count:
        raise ValueError(
            "Cannot map a generator with more channels than the `target_count`. "
            "To select a subset of the channels, use the slicing operator. "
        )
    # TODO: Maybe it would be more efficient to create an additional AGen for this
    match strategy:
        case ChannelMappingStrategy.CYCLE:
            return MultiChannelGen([gen[i % gen.channels] for i in range(target_count)])
        case ChannelMappingStrategy.LAST:
            return MultiChannelGen(
                [gen[i] for i in range(gen.channels)]
                + [gen[gen.channels - 1] for _ in range(target_count - gen.channels)]
            )
        case ChannelMappingStrategy.ZERO:
            return MultiChannelGen(
                [gen[i] for i in range(gen.channels)]
                + [0 for _ in range(target_count - gen.channels)]
            )
        case _:
            raise ValueError(
                f"Invalid mapping strategy: {strategy}. "
                f"Must be one of: {', '.join(ChannelMappingStrategy)}."
            )


class SinOsc(SingleChannelGen):
    """Simple Sine Oscillator

    Parameters
    ----------
    freq
        The frequency of the oscillator in Hz.
    amp
        The amplitude of the oscillator.
    phase
        The phase of the oscillator in radians.
    """

    def __init__(
        self,
        freq: GenOrNum,
        phase: GenOrNum = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._add_node(freq, "freq", convert_num_to_arr=True)
        self._add_node(phase, "phase")

    def _generate_single(self, sample_count: int, start: int = 0) -> np.ndarray:
        m_phase = self.state.data.get("m_phase", 0)
        # Use a cumsum here to account for varying frequencies
        phases = np.cumsum(
            np.concatenate(
                [
                    np.array([m_phase]),
                    self.nodes["freq"] / self.sr * 2.0 * np.pi,
                ]
            )
        )
        x = phases[:-1] + self.nodes["phase"]
        if x.shape[0] > 0:
            self.state.data["m_phase"] = phases[-1]
        return np.sin(x)


class WhiteNoise(SingleChannelGen):
    """WhiteNoise generator"""

    def _generate_single(self, sample_count: int, start: int) -> np.ndarray:
        return np.random.rand(sample_count) * 2 - 1


@njit(float64[:](float64, int64))
def _brown_noise(last_sample: float, sample_count: int) -> np.ndarray:
    result = np.empty(sample_count + 1)
    result[0] = last_sample
    for i in range(1, sample_count + 1):
        result[i] = result[i - 1] + np.random.rand() * 0.25 - 0.125
        if result[i] > 1:
            result[i] = 2 - result[i]
        elif result[i] < -1:
            result[i] = -2 - result[i]
    return result[1:]


class BrownNoise(SingleChannelGen):
    def _generate_single(self, sample_count, start):
        last = self.state.data.get("last", 0)
        result = _brown_noise(last, sample_count)
        self.state.data["last"] = result[-1]
        return result


class LFPulse(SingleChannelGen):
    """Non-band-limited Pulse Oscillator. output in [0,1]

    Parameters
    ----------
    freq
        the frequency of the oscillator in Hz.
    phase
        the phase of the oscillator in cycles (i.e. phase/2pi)
    width
        the duty cycle in [0, 1]
    """

    def __init__(
        self,
        freq: GenOrNum,
        phase: GenOrNum = 0.0,
        width: GenOrNum = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._add_node(freq, "freq", convert_num_to_arr=True)
        self._add_node(phase, "phase")
        self._add_node(width, "width")

    @staticmethod
    def lf_pulse_osc(phases, widths):
        return np.signbit((phases % 1.0) - widths).astype(np.float64)

    def _generate_single(self, sample_count: int, start: int) -> np.ndarray:
        m_phase = self.state.data.get("m_phase", 0)
        phases = np.cumsum(
            np.concatenate(
                [
                    np.array([m_phase]),
                    self.nodes["freq"] / self.sr,
                ]
            )
        )
        x = phases[:-1] + self.nodes["phase"]
        if x.shape[0] > 0:
            self.state.data["m_phase"] = phases[-1]
        return self.lf_pulse_osc(x, self.nodes["width"])


class LFSaw(AGen):
    """Sawtooth Oscillator: non-band-limited, range [-1, 1], starts at zero
    with positive slope.

    Parameters
    ----------
    freq
        The frequency of the oscillator in Hz.
    amp
        The amplitude of the oscillator.
    phase
        The initial (normalized) phase of the oscillator [0, 1].
    """

    def __init__(
        self,
        freq: GenOrNum,
        phase: GenOrNum = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._add_node(freq, "freq", convert_num_to_arr=True)
        self._add_node(phase, "phase")

    def _generate_new(self, sample_count: int, start: int, channel: int) -> np.ndarray:
        # Use a cumsum here to account for varying frequencies
        m_phase = self.state.data.get("m_phase", 0)
        phases = np.cumsum(
            np.concatenate(
                [
                    np.array([m_phase]),
                    self.nodes["freq"] / self.sr,
                ]
            )
        )

        x = phases[:-1] + self.nodes["phase"]
        if x.shape[0] > 0:
            self.state.data["m_phase"] = phases[-1]
        return ((x - 0.5) % 1.0) * 2 - 1


class LFTri(AGen):
    """Triangle Oscillator: non band-limited, range [-1, 1], starts at zero
    with positive slope

    Parameters
    ----------
    freq
        The frequency [Hz] of the oscillator.
    phase
        The initial (normalized) phase of the oscillator [0, 1].
    """

    def __init__(
        self,
        freq: GenOrNum,
        phase: GenOrNum = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._add_node(freq, "freq", convert_num_to_arr=True)
        self._add_node(phase, "phase")

    def _generate_new(self, sample_count: int, start: int, channel: int) -> np.ndarray:
        # Use a cumsum here to account for varying frequencies
        m_phase = self.state.data.get("m_phase", 0)
        phases = np.cumsum(
            np.concatenate(
                [
                    np.array([m_phase]),
                    self.nodes["freq"] / self.sr,
                ]
            )
        )

        x = phases[:-1] + self.nodes["phase"]
        if x.shape[0] > 0:
            self.state.data["m_phase"] = phases[-1]
        return np.abs(((x - 0.25) % 1.0) - 0.5) * 4 - 1


def klang(
    timbre: Iterable[tuple[GenOrNum, ...]],
) -> AGen:
    gens = []
    for e in timbre:
        if len(e) == 3:
            freq, amp, phase = e
        elif len(e) == 2:
            freq, amp = e
            phase = 0.0
        else:
            raise ValueError(f"Invalid timbre element: {e}")
        gens.append(amp * SinOsc(freq=freq, phase=phase))
    return AddGen(*gens)


class PlayAsig(AGen):
    """Generator for playing Asig objects.

    Note that this AGen does not support specifying a custom sample rate as it always
    uses the sample rate of the provided Asig.
    To resample the provided Asig, instead wrap this AGen with a `ResampleGen`.

    Parameters
    ----------
    asig
        The Asig object to play.
    rate
        The rate at which to play the Asig. This needs to be positive for all samples.
    loop
        Whether the Asig should be looped. This is different from using done="loop" as this would
        loop the generated signal only while this allows further modulation of the rate argument.
    """

    def __init__(
        self,
        asig: Asig | np.ndarray | str,
        rate: GenOrNum = 1,
        loop: bool = False,
        *args,
        sr: int | None = None,
        **kwargs,
    ):
        assert (
            sr is None or not isinstance(asig, Asig)
        ), "Cannot use custom sample rate here. PlayAsig always uses the sample rate of the provided AGen"
        if not isinstance(asig, Asig):
            if sr is None:
                sr = config.AUDIO_RATE
            asig = Asig(asig, sr=sr)
        
        super().__init__(
            sr=asig.sr,
            label=asig.label,
            channels=asig.channels,
            cn=asig.cn,
            **kwargs,
        )
        self.loop = loop
        self._asig = asig
        self._add_node(rate, "rate", convert_num_to_arr=True)

    def _generate_new(self, sample_count, start, channel):
        assert np.all(self.nodes["rate"] >= 0), "Rate must be positive."
        current_t = self.state.data.get("current_t", 0)
        t = np.cumsum(np.concatenate([[current_t], self.nodes["rate"]]))
        if not self.loop:
            t = t[np.ceil(t) <= self._asig.sig.shape[0]]
        else:
            t = t % self._asig.sig.shape[0]
        self.state.data["current_t"] = t[-1]
        sample_start = math.floor(np.min(t))
        sample_end = math.ceil(np.max(t))
        if self.channels == 1:
            sig = self._asig.sig[sample_start:sample_end].reshape(-1)
        else:
            sig = self._asig.sig[sample_start:sample_end, channel].reshape(-1)
        if sig.shape[0] == 0:
            return np.empty(0)
        return np.interp(
            t[:-1],
            np.arange(sample_start, sample_start + sig.shape[0]),
            sig,
        )


class Line(SingleChannelGen):
    """Generator for simple lines.

    Parameters
    ----------
    start
        The value at the start of the line.
    end
        The value at the end of the line.
    dur
        The duration of the line in seconds.
    curve
        The curvature of the line
        - curve=0: linear mapping from start to end over time
        - curve<0: curved to change faster initially, slower towards end
        - curve>0: curved to change slower initially, faster towards end
    """

    def __init__(
        self,
        start: float | int,
        end: float | int,
        dur: float | int,
        curve: float | int = 0,
        *args,
        **kwargs,
    ) -> None:
        self._start = start
        self._end = end
        self._dur = dur
        self._curve = curve
        super().__init__(*args, **kwargs)

    def get_nodes(self) -> dict[str, AGen | float | int]:
        return {
            "start": self._start,
            "end": self._end,
            "dur": self._dur,
            "curve": self._curve,
        }

    def _generate_single(self, sample_count: int, start: int) -> np.ndarray:
        start_time = start / self.sr
        end_sample = math.floor(min((start + sample_count), self._dur * self.sr))
        if abs(self._curve) < 0.001:
            slope = (self._end - self._start) / self._dur
            linear = np.linspace(
                start=self._start + slope * start_time,
                stop=self._start + slope * end_sample / self.sr,
                num=min(sample_count, max(end_sample - start, 0)),
                endpoint=False,
            )
            return linear
        else:
            tsvec = np.linspace(
                start=start_time / self._dur,
                stop=end_sample / self.sr / self._dur,
                num=min(sample_count, max(end_sample - start, 0)),
                endpoint=False,
            )
            return self._start + (self._end - self._start) / (
                1.0 - np.exp(self._curve)
            ) * (1 - np.exp(self._curve) ** tsvec)


class XLine(SingleChannelGen):
    """Exponential curve

    Parameters
    ----------
    start
        The starting value.
    end
        The ending value.
    dur
        The duration in seconds.
    """

    def __init__(
        self,
        start: float | int,
        end: float | int,
        dur: float | int,
        *args,
        **kwargs,
    ) -> None:
        self._start = start
        self._end = end
        self._dur = dur
        super().__init__(*args, **kwargs)

    def get_nodes(self) -> dict[str, AGen | float | int]:
        return {
            "start": self._start,
            "end": self._end,
            "dur": self._dur,
        }

    def _generate_single(self, sample_count: int, start: int) -> np.ndarray:
        start_time = start / self.sr
        end_sample = int(min((start + sample_count), self._dur * self.sr))
        end_time = end_sample / self.sr
        b = np.log(self._end / self._start) / self._dur
        return self._start * np.exp(
            b * np.linspace(start_time, end_time, max(0, end_sample - start))
        )


class ADSR(SingleChannelGen):
    """Generator for ADSR envelopes.

    Parameters
    ----------
    attack
        The attack time in seconds.
    decay
        The decay time in seconds.
    sustain
        The sustain time in seconds.
    release
        The release time in seconds.
    level
        The sustain level.
    """

    def __init__(
        self,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
        level: float = 0.5,
        *args,
        **kwargs,
    ) -> None:
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.level = level
        super().__init__(*args, **kwargs)

    def get_nodes(self) -> dict[str, GenOrNum]:
        return {
            "attack": self.attack,
            "decay": self.decay,
            "sustain": self.sustain,
            "release": self.release,
            "level": self.level,
        }

    def _generate_single(self, sample_count: int, start: int) -> np.ndarray:
        # TODO: This is unnecessarily complicated.

        start_time = start / self.sr
        end_time = (start + sample_count) / self.sr

        attack_samples = int(self.attack * self.sr)
        decay_samples = int(self.decay * self.sr)
        sustain_samples = int(self.sustain * self.sr)
        release_samples = int(self.release * self.sr)

        attack_end_sample = min(max(attack_samples - start, 0), sample_count)

        attack_slope = 1 / self.attack
        attack = np.linspace(
            start=attack_slope * start_time,
            stop=min(1, end_time * attack_slope),
            num=attack_end_sample,
        )

        decay_slope = (self.level - 1) / self.decay
        decay_end_sample = min(
            max(attack_samples - start + decay_samples, 0), sample_count
        )

        decay = np.linspace(
            start=min(1, 1 + decay_slope * (start - attack_samples) / self.sr),
            stop=max(
                self.level,
                1
                + decay_slope
                * ((decay_end_sample - (attack_samples - start)) / self.sr),
            ),
            num=decay_end_sample - attack_end_sample,
        )

        sustain_end_sample = min(
            max(attack_samples - start + decay_samples + sustain_samples, 0),
            sample_count,
        )
        sustain = np.full(
            shape=sustain_end_sample - decay_end_sample, fill_value=self.level
        )

        release_slope = -self.level / self.release
        release_end_sample = min(
            max(
                attack_samples
                - start
                + decay_samples
                + sustain_samples
                + release_samples,
                0,
            ),
            sample_count,
        )
        release = np.linspace(
            start=min(
                self.level,
                self.level
                + release_slope
                * (start - attack_samples - decay_samples - sustain_samples)
                / self.sr,
            ),
            stop=max(
                0,
                self.level
                + release_slope
                * (
                    release_end_sample
                    - (attack_samples + decay_samples + sustain_samples - start)
                )
                / self.sr,
            ),
            num=release_end_sample - sustain_end_sample,
        )

        return np.concatenate(
            [
                attack,
                decay,
                sustain,
                release,
            ]
        )


class Env(SingleChannelGen):
    """Env Envelope

    Parameters
    ----------
    values
        The array of values
    dtimes
        The array of time deltas dtime between values
    """

    def __init__(
        self,
        values: list[float] | np.ndarray,
        dtimes: list[float] | np.ndarray | float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._values = np.array(values)
        
        if isinstance(dtimes, float) or isinstance(dtimes, int):
            self._dtimes = np.full((values.shape[0] - 1,), dtimes)
        else:
            self._dtimes = np.array(dtimes)
        self._times = np.concatenate((np.zeros(1), self.sr * np.cumsum(self._dtimes)))

    def get_nodes(self) -> dict[str, GenOrNum]:
        return {
            # TODO: How should this be represented in the graph?
            # The arrays could potentially be very large.
        }

    def _generate_single(self, sample_count: int, start: int) -> np.ndarray:
        end_index = min(start + sample_count, int(self._times[-1]))

        tnew = np.linspace(start, end_index, end_index - start, endpoint=False)
        ynew = np.interp(tnew, self._times, self._values)
        return ynew


class LFilter(SingleChannelGen):
    """AGen for filtering signals with IIR or FIR filters.

    This generator is a wrapper around the `scipy.signal.lfilter` function.

    Parameters
    ----------
    gen
        The input signal to filter.
    a
        The denominator coefficients of the filter.
    b
        The numerator coefficients of the filter.

    For more information on the parameters, see the documentation of `scipy.signal.lfilter`.
    """

    def __init__(self, gen: GenOrNum, a: list[float], b: list[float], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_node(gen, "gen", convert_num_to_arr=True)
        self.a = a
        self.b = b

    def _generate_single(self, sample_count, start):
        z = self.state.data.get("z")
        if z is None:
            z = lfiltic(self.b, self.a, [], [])
        y, z = lfilter(self.b, self.a, self.nodes["gen"], zi=z)
        self.state.data["z"] = z
        return y


def digisinc(x: np.ndarray, M: np.ndarray) -> np.ndarray:
    denom = M * np.sin(np.pi * x)
    num = np.sin(np.pi * x * M)
    filt = np.abs(denom) > 0.0005
    res = np.empty_like(x, dtype=float)
    res[filt] = num[filt] / denom[filt] - 1 / M[filt]
    res[~filt] = 1
    return res


class BLIT(AGen):
    """Band-Limited Impulse Train.

    Implementation of https://www.music.mcgill.ca/~gary/307/week5/node14.html

    Parameters
    ----------
    freq
        The frequency of the impulse train.
    even
        Whether the number of harmonics should be even.
    m
        The maximum number of harmonics. If not provided, it is calculated as `sr * 0.5 / freq`.
        If this value is too high, the number of harmonics is clipped to avoid aliasing.
    """

    def __init__(
        self,
        freq: GenOrNum,
        even: bool = False,
        m: GenOrNum | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._add_node(freq, "freq", convert_num_to_arr=True)
        if m is not None:
            self._add_node(m, "m")
        self.even = even

    def get_nodes(self) -> dict[str, GenOrNum]:
        return {"even": self.even, **super().get_nodes()}

    def _generate_new(self, sample_count, start, channel):
        m_phase = self.state.data.get("m_phase", 0)
        # Use a cumsum here to account for varying frequencies
        phases = np.cumsum(
            np.concatenate([np.array([m_phase]), self.nodes["freq"] / self.sr])
        )
        x = phases[:-1]
        freq = self.nodes["freq"]
        if x.shape[0] > 0:
            self.state.data["m_phase"] = phases[-1]

        period = self.sr / freq
        M = np.floor(period).astype(int)
        even_filt = M % 2 == 0
        if self.even:
            M[~even_filt] -= 1
        else:
            M[even_filt] -= 1
        if self.nodes.get("m") is not None:
            M = np.minimum(M, np.array(self.nodes["m"], dtype=int))

        return digisinc(x, M)

class BLImp(BLIT):
    """Band-Limited Impulse generator. Similar to Blip in SuperCollider. 
    
    Parameters
    ----------
    freq
        The frequency of the impulses in Hz. 
    numharm
        The number of harmonics. 
    """

    def __init__(self, freq, numharm, *args, **kwargs):
        if isinstance(numharm, AGen):
            n_floor = numharm.apply(np.floor)
        else:
            n_floor = math.floor(numharm)
        self.numharm = numharm
        self.freq = freq
        super().__init__(freq, even=False, m=n_floor * 2 + 1, *args, **kwargs)

    def get_nodes(self):
        return {
            "numharm": self.numharm, 
            "freq": self.freq, 
        }


class BLSaw(AGen):
    """Band-Limited Sawtooth Oscillator.

    This oscillator applies a leaky integrator to a BLIT signal.

    Parameters
    ----------
    freq
        The frequency of the oscillator in Hz.
    even
        Whether the number of harmonics should be even.
    m
        The maximum number of harmonics. If not provided, it is calculated as `sr * 0.5 / freq`.
        If this value is too high, the number of harmonics is clipped to avoid aliasing.
    """

    def __init__(
        self,
        freq: GenOrNum,
        even: bool = False,
        m: GenOrNum | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._add_node(
            LFilter(BLIT(freq=freq, even=even, m=m), a=[1, -0.999], b=[1]), "saw"
        )
        self.freq = freq
        self.even = even
        self.m = m

    def get_nodes(self) -> dict[str, GenOrNum]:
        return {
            "freq": self.freq,
            "even": self.even,
            "m": self.m,
        }

    def _generate_new(self, sample_count, start, channel) -> np.ndarray:
        return self.nodes["saw"]


class BLPulse(SingleChannelGen):
    """Band-Limited Pulse Oscillator

    This oscillator applies a leaky integrator to a bi-polar BLIT signal, as described in
    https://www.music.mcgill.ca/~gary/307/week5/node15.html.

    Parameters
    ----------
    freq
        The frequency of the oscillator in Hz.
    width
        The width of the pulse in [0, 1].
    """

    def __init__(self, freq: GenOrNum, width: GenOrNum = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_node(freq, "freq", convert_num_to_arr=True)
        self._add_node(width, "width")

    def _generate_single(self, sample_count, start):
        m_phase = self.state.data.get("m_phase", 0)
        # Use a cumsum here to account for varying frequencies
        phases = np.cumsum(
            np.concatenate([np.array([m_phase]), self.nodes["freq"] / self.sr])
        )
        x = phases[:-1]
        freq = self.nodes["freq"]
        self.state.data["m_phase"] = phases[-1]

        period = self.sr / freq
        M = np.floor(period).astype(int)
        even_filt = M % 2 == 0
        M[even_filt] -= 1
        if self.nodes.get("m") is not None:
            M = np.minimum(M, self.nodes["m"].astype(int))

        blit_1 = digisinc(x, M)
        blit_2 = digisinc(x + self.nodes["width"], M)

        z = self.state.data.get("z")
        if z is None:
            z = lfiltic([1], [1, -0.999], [], [])
        y, z = lfilter([1], [1, -0.999], blit_1 - blit_2, zi=z)
        self.state.data["z"] = z
        return y


class LoopGen(AGen):
    """Generator for looping other generators in a specified range.

    Parameters
    ----------
    gen
        The generator to loop.
    loop_start
        Where to start the loop in seconds. If the generator terminates before this point,
        the LoopGen will terminate as well.
    loop_end
        Where to end the loop in seconds. If not provided, this will be determined by
        the duration of the generator. If the generator does not terminate, it will not
        be looped in this case.
    """

    # TODO: Add rate control argument
    def __init__(
        self,
        gen: AGen,
        loop_start: float | None = None,
        start_trigger: GenOrNum | None = None,
        end_trigger: GenOrNum | None = None,
        loop_end: float | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert loop_start is not None or start_trigger is not None
        if start_trigger is not None:
            self._add_node(start_trigger, "start")
        if end_trigger is not None:
            self._add_node(end_trigger, "end")
        self.gen = gen
        self.loop_start = loop_start
        self.loop_end = loop_end

    @staticmethod
    def _get_trigger_index(buf: np.ndarray) -> int | None:
        res = np.argwhere(buf >= 1).flatten()
        if res.shape[0] == 0:
            return None
        return res[0]

    def get_nodes(self):
        return {
            "loop_start": self.loop_start,
            "loop_end": self.loop_end,
            "gen": self.gen,
            **super().get_nodes(),
        }

    def _generate_new(self, sample_count, start, channel):
        loop_start_sample = self.state.data.get(
            "start",
            int(self.loop_start * self.sr) if self.loop_start is not None else None,
        )
        if loop_start_sample is None:
            loop_start_sample = self._get_trigger_index(self.nodes["start"])
            if loop_start_sample is not None:
                loop_start_sample += start
            self.state.data["start"] = loop_start_sample
        if loop_start_sample is None or loop_start_sample > start + sample_count:
            return self._get_samples(self.gen, sample_count, start, channel)

        unlooped_count = max(0, loop_start_sample - start)
        if unlooped_count > 0:
            unlooped = self._get_samples(self.gen, unlooped_count, start, channel)
            if unlooped.shape[0] < unlooped_count:
                # Generator has teminated before the loop start, so terminate
                return unlooped
        else:
            unlooped = np.array([])

        loop_end = self.state.data.get(
            "loop_end",
            math.floor(self.loop_end * self.sr) if self.loop_end is not None else None,
        )

        start = (
            (start + unlooped.shape[0] - loop_start_sample)
            % (loop_end - loop_start_sample)
            + loop_start_sample
            if loop_end is not None
            else start + unlooped.shape[0]
        )

        looped_len = (
            loop_end - start
            if loop_end is not None
            else sample_count - unlooped.shape[0]
        )

        looped = self._get_samples(self.gen, looped_len, start, channel=channel)

        if looped.shape[0] < looped_len:
            # The the generator has terminated, update the loop_end
            loop_end = start + unlooped.shape[0] + looped.shape[0]
            self.state.data["loop_end"] = loop_end
            looped_len = self.state.data["loop_end"] - loop_start_sample

        if unlooped.shape[0] + looped.shape[0] >= sample_count:
            # If unlooped + looped is enough to fill the sample_count, return
            return np.concatenate([unlooped, looped])[:sample_count]
        return np.concatenate(
            [
                unlooped,
                looped,
                np.tile(
                    # Start from the beginning for the remaining repetitions
                    self._get_samples(
                        self.gen,
                        loop_end - loop_start_sample,
                        loop_start_sample,
                        channel=channel,
                    ),
                    math.ceil(
                        (sample_count - looped.shape[0] - unlooped.shape[0])
                        / looped_len
                    ),
                ),
            ]
        )[:sample_count]


class Pan2(AGen):
    """Two-channel panner.

    Parameters
    ----------
    gen
        The input signal. This AGen should have either one or two channels.
    pos
        The position of the panner in the range [-1, 1]. Values outside of this range will
        be clipped. This could in theory also have either one or two channels.
    """

    def __init__(self, gen: GenOrNum, pos: GenOrNum, *args, **kwargs):
        super().__init__(*args, channels=2, **kwargs)
        self._add_node(gen, "gen", convert_num_to_arr=True)
        self._add_node(pos, "pos", convert_num_to_arr=True)

    def _generate_new(self, sample_count, start, channel):
        pos = (np.clip(self.nodes["pos"], -1, 1) + 1) * np.pi / 4
        if channel == 0:
            return np.cos(pos) * self.nodes["gen"]
        else:
            return np.sin(pos) * self.nodes["gen"]


class PanAz(AGen):
    """Azimuth panner. Similar to PanAz in SuperCollider.

    Parameters
    ----------
    channels
        The number of output channels. All other arguments should either have the same
        number of channels or be single-channel.
    gen
        The input signal.
    pos
        The position of the panner in the range [-1, 1]. Values outside of this range will
        be clipped.
    width
        The width of the panning envelope.
    orientation
        The base orientation. E.g. 0.0 if one speaker is in front.
    """

    def __init__(
        self,
        channels: int,
        gen: GenOrNum,
        pos: GenOrNum,
        width: GenOrNum = 2.0,
        orientation: GenOrNum = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, channels=channels, **kwargs)
        self._add_node(gen, "gen", convert_num_to_arr=True)
        self._add_node(pos, "pos")
        self._add_node(width, "width")
        self._add_node(orientation, "orientation")

    def _generate_new(self, sample_count, start, channel):
        pos = np.clip(self.nodes["pos"], -1, 1)
        width = self.nodes["width"]
        orientation = np.clip(self.nodes["orientation"], -1, 1)
        x = self.nodes["gen"]

        pos = pos * 0.5 * self.channels + width * 0.5 + orientation
        chanpos = (pos - channel + 1) / width
        # Map negative numbers to the equivalent positive number
        chanpos %= self.channels / width
        return x * np.sin(np.pi * np.clip(chanpos, 0, 1))


class TimeMode(str, Enum):
    """Enum representing time units."""

    SAMPLES = "samples"
    """Time in samples."""

    SECONDS = "seconds"
    """Time in seconds."""


class SeqAGen(AGen):
    """Sequence of generators.

    Parameters
    ----------
    gens
        A list containing tuples with the onset time and AGen.
    time_mode
        The unit of the onset times. Either "seconds" or "samples".
    """

    def __init__(
        self,
        gens: Sequence[tuple[float, GenOrNum]],
        time_mode: TimeMode = TimeMode.SECONDS,
        *args,
        **kwargs,
    ):
        if time_mode not in [a.value for a in TimeMode]:
            raise ValueError(
                f"Invalid time_mode: {time_mode}. Must be one of: {', '.join(TimeMode)}."
            )
        gen_values = [gen for _, gen in gens]
        super().__init__(
            *args,
            sr=get_max_sr(gen_values),
            channels=get_max_channel(gen_values),
            **kwargs,
        )
        self.gens = sorted(gens, key=lambda x: x[0])
        self.time_mode = time_mode

    def create_graph(self, additional_attr=[]):
        if len(self.gens) > 5 and config.PLOT_SEQUENCE_AS_GRAPH_WARNING:
            warnings.warn(
                "Plotting a long sequence of generators may yield a very large and "
                "unreadable graph. "
                "You can use the `plot_sequence` to plot a timeline of the sequence "
                "instead."
            )
        return super().create_graph(additional_attr)

    def get_nodes(self):
        return {
            **{f"Onset: {onset}": gen for onset, gen in self.gens},
            **super().get_nodes(),
        }

    def plot_sequence(self, ax=None):
        """Plots the sequence of generators on a timeline."""
        import matplotlib.pyplot as plt

        ax = plt.gca() or ax
        onsets = np.array([onset for onset, _ in self.gens], dtype=float)
        if self.time_mode == TimeMode.SAMPLES:
            onsets /= self.sr
        ax.vlines(onsets, 0, 0.5 * ((np.arange(len(self.gens)) % 2 == 0) * 2 - 1))
        ax.axvline(0, c="gray", lw=0.5)
        ax.axhline(0, c="black")
        ax.plot(onsets, np.zeros_like(self.gens), "ko", mfc="white")
        ax.set_xlabel("Time (Seconds)")
        ax.set_ylim(-1, 1)
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        ax.set_xlim(-1, np.max(onsets) + 1)
        for i, (_, gen) in enumerate(self.gens):
            ax.annotate(
                gen.label if isinstance(gen, AGen) else str(gen),
                (onsets[i], 0.5 if i % 2 == 0 else -0.5),
                textcoords="offset points",
                verticalalignment="bottom" if i % 2 == 0 else "top",
                xytext=(-2, 2 if i % 2 == 0 else -2),
                bbox=dict(boxstyle="square", pad=0, lw=0, fc=(1, 1, 1, 0.7)),
            )

    def _generate_new(self, sample_count, start, channel):
        result = np.zeros(sample_count)
        max_len = 0
        for onset, gen in self.gens:
            onset = max(onset, 0)
            onset_sample = (
                int(onset * self.sr) if self.time_mode == TimeMode.SECONDS else onset
            )
            if onset_sample >= start + sample_count:
                return result
            new_samples: np.ndarray = self._get_samples(
                gen,
                sample_count - max(onset_sample - start, 0),
                max(start - onset_sample, 0),
                channel,
                convert_num_to_array=True,
            )  # type: ignore
            max_len = max(max_len, new_samples.shape[0] + max(onset_sample - start, 0))
            result[
                max(onset_sample - start, 0) : new_samples.shape[0]
                + max(onset_sample - start, 0)
            ] += new_samples
        return result[:max_len]
    

# TH: proposal to replace by more flexible Resample
#     or to integrate that even deeper in core AGen._get_samples() as resampling default
# remove old ResampleGen temporarily 

# class ResampleGen(SingleChannelGen):
#     """Resamples a given AGen.

#     This can be useful to resample an AGen where the sample rate cannot be customized,
#     such as `PlayAsig`.

#     Parameters
#     ----------
#     gen
#         The AGen that should be resampled.
#     """

#     def __init__(self, gen: GenOrNum, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._add_node(gen, "gen", convert_num_to_arr=True)

#     def get_nodes(self):
#         return {"sr": self.sr, **super().get_nodes()}

#     def _generate_single(self, sample_count, start):
#         return self.nodes["gen"]


@njit(
    Tuple((float64[:], float64, float64, float64, float64))(
        float64[:], float64[:], float64, float64, float64, float64, float64
    )
)
def _lpf_numba(
    x: np.ndarray,
    f_0: np.ndarray,
    sr: float,
    y_1: float,
    y_2: float,
    x_1: float,
    x_2: float,
):
    # https://stackoverflow.com/a/20932062
    c = 1.0 / np.tan(np.pi * f_0 / sr)
    c_2 = c * c
    q = np.sqrt(2.0)
    b_0 = 1.0 / (1.0 + q * c + c_2)
    b_1 = 2.0 * b_0
    b_2 = b_0
    a_1 = 2.0 * b_0 * (1.0 - c_2)
    a_2 = b_0 * (1.0 - q * c + c_2)
    result = np.empty_like(x)
    for i in range(len(x)):
        result[i] = (
            b_0[i] * x[i] + b_1[i] * x_1 + b_2[i] * x_2 - a_1[i] * y_1 - a_2[i] * y_2
        )
        x_2 = x_1
        x_1 = x[i]
        y_2 = y_1
        y_1 = result[i]
    return result, y_1, y_2, x_1, x_2


class LPF(SingleChannelGen):
    """Low-pass filter."""

    def __init__(self, gen: GenOrNum, freq: GenOrNum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_node(freq, "freq", convert_num_to_arr=True)
        self._add_node(gen, "gen", convert_num_to_arr=True)

    def _generate_single(self, sample_count, start):
        y_1, y_2, x_1, x_2 = self.state.data.get("z", (0, 0, 0, 0))
        out, y_1, y_2, x_1, x_2 = _lpf_numba(
            self.nodes["gen"],  # type: ignore
            self.nodes["freq"],  # type: ignore
            self.sr,
            y_1,
            y_2,
            x_1,
            x_2,
        )
        self.state.data["z"] = y_1, y_2, x_1, x_2
        return out


@njit(
    Tuple((float64[:], float64, float64, float64, float64))(
        float64[:], float64[:], float64, float64, float64, float64, float64
    )
)
def _hpf_numba(
    x: np.ndarray,
    f_0: np.ndarray,
    sr: float,
    y_1: float,
    y_2: float,
    x_1: float,
    x_2: float,
):
    # https://stackoverflow.com/a/39240059
    c = 1.0 / np.tan(np.pi * f_0 / sr)
    q = np.sqrt(2.0)
    c_2 = c * c
    b_0 = 1.0 / (1.0 + q * c + c_2)
    b_1 = 2.0 * b_0
    b_2 = b_0
    a_1 = 2.0 * b_0 * (1.0 - c_2)
    a_2 = b_0 * (1.0 - q * c + c_2)

    result = np.empty_like(x)
    for i in range(len(x)):
        result[i] = (
            b_0[i] * c_2[i] * x[i]
            - b_1[i] * c_2[i] * x_1
            + b_2[i] * c_2[i] * x_2
            - a_1[i] * y_1
            - a_2[i] * y_2
        )
        x_2 = x_1
        x_1 = x[i]
        y_2 = y_1
        y_1 = result[i]
    return result, y_1, y_2, x_1, x_2


class HPF(SingleChannelGen):
    """High-pass filter."""

    def __init__(self, gen: GenOrNum, freq: GenOrNum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_node(freq, "freq", convert_num_to_arr=True)
        self._add_node(gen, "gen", convert_num_to_arr=True)

    def _generate_single(self, sample_count, start):
        y_1, y_2, x_1, x_2 = self.state.data.get("z", (0, 0, 0, 0))
        out, y_1, y_2, x_1, x_2 = _hpf_numba(
            self.nodes["gen"],
            self.nodes["freq"],
            self.sr,
            y_1,
            y_2,
            x_1,
            x_2,
        )
        self.state.data["z"] = y_1, y_2, x_1, x_2
        return out


@njit(
    Tuple((float64[:], float64, float64, float64, float64))(
        float64[:], float64[:], float64[:], float64, float64, float64, float64, float64
    )
)
def _brf_numba(
    x: np.ndarray,
    f_0: np.ndarray,
    bw: np.ndarray,
    sr: float,
    y_1: float,
    y_2: float,
    x_1: float,
    x_2: float,
):
    # See https://www.w3.org/TR/audio-eq-cookbook/ for more information
    omega_0 = 2.0 * np.pi * f_0 / sr
    sin_omega_0 = np.sin(omega_0)
    cos_omega_0 = np.cos(omega_0)
    alpha = sin_omega_0 * np.sinh(np.log(2) / 2.0 * bw * omega_0 / sin_omega_0)
    result = np.zeros_like(x)

    a_0 = 1.0 + alpha
    a_1 = -2 * cos_omega_0
    a_2 = 1.0 - alpha
    b_1 = -2 * cos_omega_0

    for i in range(len(x)):
        result[i] = (
            1.0 / a_0[i] * (x[i] + x_2 + x_1 * b_1[i] - a_1[i] * y_1 - a_2[i] * y_2)
        )
        x_2 = x_1
        x_1 = x[i]
        y_2 = y_1
        y_1 = result[i]

    return result, y_1, y_2, x_1, x_2


class BRF(SingleChannelGen):
    """Band-reject filter."""

    def __init__(self, gen, f_0: GenOrNum, bw: GenOrNum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_node(f_0, "f_0", convert_num_to_arr=True)
        self._add_node(bw, "bw", convert_num_to_arr=True)
        self._add_node(gen, "gen", convert_num_to_arr=True)

    def _generate_single(self, sample_count, start):
        y_1, y_2, x_1, x_2 = self.state.data.get("z", (0, 0, 0, 0))
        out, y_1, y_2, x_1, x_2 = _brf_numba(
            self.nodes["gen"],
            self.nodes["f_0"],
            self.nodes["bw"],
            self.sr,
            y_1,
            y_2,
            x_1,
            x_2,
        )
        self.state.data["z"] = y_1, y_2, x_1, x_2
        return out


@njit(
    Tuple((float64[:], float64, float64, float64, float64))(
        float64[:], float64[:], float64[:], float64, float64, float64, float64, float64
    )
)
def _bpf_numba(
    x: np.ndarray,
    f_0: np.ndarray,
    bw: np.ndarray,
    sr: float,
    y_1: float,
    y_2: float,
    x_1: float,
    x_2: float,
):
    # See https://www.w3.org/TR/audio-eq-cookbook/ for more information
    omega_0 = 2.0 * np.pi * f_0 / sr
    sin_omega_0 = np.sin(omega_0)
    alpha = sin_omega_0 * np.sinh(np.log(2) / 2.0 * bw * omega_0 / sin_omega_0)
    result = np.zeros_like(x)

    a_0 = 1.0 + alpha
    a_1 = -2 * np.cos(omega_0)
    a_2 = 1.0 - alpha
    b_0 = alpha
    b_2 = -alpha

    for i in range(len(x)):
        result[i] = (
            1.0 / a_0[i] * (b_0[i] * x[i] + b_2[i] * x_2 - a_1[i] * y_1 - a_2[i] * y_2)
        )
        x_2 = x_1
        x_1 = x[i]
        y_2 = y_1
        y_1 = result[i]

    return result, y_1, y_2, x_1, x_2


class BPF(SingleChannelGen):
    """Band-pass filter."""

    def __init__(self, gen, f_0: GenOrNum, bw: GenOrNum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_node(f_0, "f_0", convert_num_to_arr=True)
        self._add_node(bw, "bw", convert_num_to_arr=True)
        self._add_node(gen, "gen", convert_num_to_arr=True)

    def _generate_single(self, sample_count, start):
        y_1, y_2, x_1, x_2 = self.state.data.get("z", (0, 0, 0, 0))
        out, y_1, y_2, x_1, x_2 = _bpf_numba(
            self.nodes["gen"],
            self.nodes["f_0"],
            self.nodes["bw"],
            self.sr,
            y_1,
            y_2,
            x_1,
            x_2,
        )
        self.state.data["z"] = y_1, y_2, x_1, x_2
        return out


class OneZero(SingleChannelGen):
    """One-zero filter."""

    def __init__(self, gen: GenOrNum, coef: GenOrNum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_node(gen, "gen", convert_num_to_arr=True)
        self._add_node(coef, "coef")

    def _generate_single(self, sample_count, start):
        last = self.state.data.get("last", 0)
        coef = self.nodes["coef"]
        gen = self.nodes["gen"]
        result = gen * (1 - np.abs(coef)) + np.concatenate([[last], gen[:-1]]) * coef
        self.state.data["last"] = gen[-1]
        return result


@njit(Tuple((float64[:], float64))(float64[:], float64[:], float64))
def _one_pole_numba(
    x: np.ndarray,
    coef: np.ndarray,
    y_1: float,
) -> tuple[np.ndarray, float]:
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = (1 - abs(coef[i])) * x[i] + coef[i] * y_1
        y_1 = out[i]
    return out, y_1


class OnePole(SingleChannelGen):
    """One-pole filter."""

    def __init__(self, gen: GenOrNum, coef: GenOrNum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_node(coef, "coef", convert_num_to_arr=True)
        self._add_node(gen, "gen", convert_num_to_arr=True)

    def _generate_single(self, sample_count, start):
        y_1 = self.state.data.get("y_1", 0.0)
        gen = self.nodes["gen"]
        coef = self.nodes["coef"]
        out, y_1 = _one_pole_numba(gen, coef, y_1)
        self.state.data["y_1"] = y_1
        return out


@njit(Tuple((float64[:], float64))(float64[:], int64, float64))
def _pluck_numba(
    wavetable: np.ndarray, 
    sample_count: int, 
    last: float, 
) -> tuple[np.ndarray, float]:
    result = np.zeros(sample_count, dtype=np.float64)
    for i in range(sample_count):
        current_sample = i % len(wavetable)
        wavetable[current_sample] = 0.5 * (wavetable[current_sample] + last)
        result[i] = wavetable[current_sample]
        last = wavetable[current_sample]
    return result, last


class Pluck(AGen):
    """A Karplus-Strong plucked string synthesis generator.
    
    Based on the implementation in https://flothesof.github.io/Karplus-Strong-algorithm-Python.html.

    Parameters
    ----------
    freq
        The fundamental frequency of the plucked string in Hz. Note that currently, this cannot 
        be modulated.
    """

    def __init__(self, freq: float | int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        w_size = self.sr // int(freq)
        self.wavetable = (2 * np.random.randint(0, 2, w_size) - 1).astype(np.float64)

    def _generate_new(self, sample_count, start, channel):
        result = np.zeros(sample_count, dtype=np.float64)
        last = self.state.data.get("last", 0)
        result, last = _pluck_numba(self.wavetable, sample_count, last)
        self.state.data["last"] = last
        return result