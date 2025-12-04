from __future__ import annotations

import math
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import pyamapping as pam
from pya.asig import Asig

from pya.agen import config

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable
    import graphviz
    from pya.agen.types import GenOrNum


class AGenState:
    def __init__(self) -> None:
        self.__samples: np.ndarray = np.empty(0)
        self.__length: int = 0
        self.__finished: bool = False

        self.data: dict[str, Any] = {}
        """Custom state data for the generator."""

    def mark_finished(self) -> None:
        self.__finished = True

    @property
    def finished(self) -> bool:
        """Whether the generator is done generating samples."""
        return self.__finished

    @property
    def length(self) -> int:
        """The current length of the cache."""
        return self.__length

    def add_to_cache(self, samples: np.ndarray) -> None:
        new_len = self.__length + samples.shape[0]
        if new_len > self.__samples.shape[0]:
            if new_len > self.__samples.shape[0] * 2:
                self.__samples = np.resize(self.__samples, new_len)
            else:
                self.__samples = np.concatenate(
                    [
                        self.__samples,
                        np.empty_like(self.__samples),
                    ],
                    axis=0,
                )
        self.__samples[self.__length : new_len] = samples
        self.__length = new_len

    def get_from_cache(self, sample_count: int, start: int) -> np.ndarray:
        if start + sample_count > self.__length and not self.__finished:
            raise ValueError("Requested samples are not in cache")
        view = self.__samples[
            min(start, self.__length) : min(start + sample_count, self.__length)
        ]
        view.flags.writeable = False
        return view


class PaddingType(str, Enum):
    """Enum for different strategies of padding values for delayed generators."""

    ZERO = "zero"
    """Pad values with zeros."""

    FIRST = "first"
    """Pad values with the first value of the generator."""


class DoneAction(str, Enum):
    """Enum for different actions to perform when a generator is done generating samples."""

    STOP = "stop"
    """Stop generating samples and return the generated samples.
    This will propagate up the tree of generators. If the generator has no parents with
    either `LAST` or `LOOP` done actions, the whole tree will terminate.
    """

    LAST = "last"
    """Repeat the last generated sample."""

    LOOP = "loop"
    """Loop the generator from the beginning."""

    ZERO = "zero"
    """Fill with zeros."""


class AGen(ABC):
    """Abstract base class for audio generators.

    Parameters
    ----------
    sr
        The sample rate of the generator. If `None`, the sample rate will be set to
        the maximum sample rate of the child generators.
    label
        A label for the generator which specifies the label of the generated Asig.
    channels
        The number of channels of the generator.
    cn
        The channel names of the generator.
    done
        The action to perform when the generator is done generating samples.
    """

    class Node:
        def __init__(self, gen: GenOrNum, convert_num_to_arr: bool):
            self.gen = gen
            self.convert_num_to_arr = convert_num_to_arr
            self.value: float | np.ndarray | None = None

    # region - MAGIC -
    def __init__(
        self,
        label: str | None = None,
        channels: int = 1,
        cn: list | None = None,
        *,
        sr: int | None = 44100,
        done: DoneAction | str = DoneAction.STOP,
        downsample_children: bool = False,
    ) -> None:
        self.nodes: dict[str, float | int | np.ndarray] = {}
        self._node_items: dict[str, AGen.Node] = {}

        self.cache: dict[int, np.ndarray] = {}

        self.__adaptive_sr = sr is None

        if sr is None:
            sr = -1
        self.sr = sr

        if label is None:
            self.label = self.__class__.__name__
        else:
            self.label = label

        self.channels = channels
        self.cn = cn
        if done not in [a.value for a in DoneAction]:
            raise ValueError(
                f"Invalid done action: {done}. Must be one of {', '.join(DoneAction)}"
            )
        self.done = done
        self.downsample_children = downsample_children
        self.states = None
        self.state = None  # type: ignore

        self.uuid = uuid.uuid4()

    def __sub__(self, other: GenOrNum) -> AGen:
        return AddGen(self, -other)

    def __rsub__(self, other: GenOrNum) -> AGen:
        return AddGen(other, -self)

    def __neg__(self) -> AGen:
        return MulGen(self, -1)

    def __add__(self, other: GenOrNum) -> AGen:
        return AddGen(self, other)

    def __radd__(self, other: GenOrNum) -> AGen:
        return AddGen(other, self)

    def __mul__(self, other: GenOrNum) -> AGen:
        return MulGen(self, other)

    def __rmul__(self, other: GenOrNum) -> AGen:
        return MulGen(other, self)

    def __pow__(self, other: GenOrNum) -> AGen:
        return PowGen(self, other)

    def __rpow__(self, other: GenOrNum) -> AGen:
        return PowGen(other, self)

    def __truediv__(self, other: GenOrNum):
        return DivGen(self, other)

    def __rtruediv__(self, other: GenOrNum):
        return DivGen(other, self)

    def __and__(self, other: GenOrNum) -> ConcatGen:
        return ConcatGen(self, other)

    def __rand__(self, other: GenOrNum) -> ConcatGen:
        return ConcatGen(other, self)

    def __getitem__(
        self,
        index: int | slice | list[bool] | list[int],
    ) -> ChannelSelectorGen:
        return ChannelSelectorGen(self, index)

    # endregion

    # region - PROTECTED -
    def _add_node(
        self,
        node: GenOrNum,
        name: str,
        convert_num_to_arr: bool = False,
    ) -> AGen.Node:
        """Add a node to the generator.
        This node will be sampled before _generate_new is called.

        Parameters
        ----------
        node
            The generator or number to add as a node.
        name
            The name of the node.
        convert_num_to_arr
            Whether to convert numbers to numpy arrays.

        Returns
        -------
        AGen.Node
            The added node.
        """
        if isinstance(node, AGen.Node):
            c = node
        else:
            c = self.Node(node, convert_num_to_arr)
    
        self._node_items[name] = c
        if isinstance(node, AGen):
            if self.__adaptive_sr:
                self.sr = max(self.sr, node.sr)
            if node.channels > 1:
                if self.channels > 1 and node.channels != self.channels:
                    raise ValueError(
                        "Cannot combine AGen with m channels with AGen with n channels "
                        "where m != n; n, m > 1."
                    )
                else:
                    self.channels = node.channels
                    self.cn = node.cn
        return c

    def __prepare_nodes(self, sample_count: int, start: int, channel: int) -> None:
        samples = {
            name: self._get_samples(
                node.gen,
                sample_count,
                start,
                channel=channel,
                convert_num_to_array=node.convert_num_to_arr,
            )
            for name, node in self._node_items.items()
        }

        min_len = min(
            (len(s) for s in samples.values() if isinstance(s, np.ndarray)),
            default=sample_count,
        )

        for name, data in samples.items():
            if isinstance(data, np.ndarray):
                samples = data[:min_len]
            else:
                samples = data
            self.nodes[name] = samples
            self._node_items[name].value = samples

    def __clear_nodes(self) -> None:
        self.nodes.clear()
        for node in self._node_items.values():
            node.value = None

    @abstractmethod
    def _generate_new(self, sample_count: int, start: int, channel: int) -> np.ndarray:
        """Generate new samples for the generator.

        This method is not called when the samples are already cached.

        Parameters
        ----------
        sample_count
            The number of samples to generate.
        start
            The starting sample index.

        Returns
        -------
        np.ndarray
            The generated samples.
        """

    def _get_samples(
        self,
        generator: GenOrNum,
        sample_count: int,
        start: int,
        channel: int,
        convert_num_to_array: bool = False,
    ) -> np.ndarray | float:
        if isinstance(generator, AGen):
            if generator.sr == self.sr:
                return generator.generate(
                    sample_count,
                    start,
                    channel=channel if generator.channels > 1 else 0,
                )
            else:
                if config.UPSAMPLE_WARNING:
                    warnings.warn(
                        "Upsampling is currently experimental and does not always produce expected results."
                    )
                if generator.sr > self.sr and not self.downsample_children:
                    raise ValueError(
                        "Child of AGen cannot have a higher sampling rate than the parent AGen.\n"
                        "If you wish to downsample the child to match the parent's sampling rate, "
                        "pass `downsample_children=True` to the constructor of the parent AGen."
                    )
                if max(self.sr, generator.sr) % min(self.sr, generator.sr) != 0:
                    raise ValueError(
                        "Target sample rate must divisible by source sample rate"
                    )
                new_start = math.floor(start * (generator.sr / self.sr))
                new_sample_count = (
                    math.floor(sample_count * (generator.sr / self.sr)) + 1
                )
                samples = generator.generate(
                    new_sample_count,
                    new_start,
                    channel=channel,
                )

                if samples.shape[0] <= 1:
                    # Return empty array as we do not have enough samples to interpolate
                    return np.array([])
                if samples.shape[0] < new_sample_count:
                    # upsampled_count = samples.shape[0] * self.sr // generator.sr - 1
                    result = np.interp(
                        np.linspace(
                            start,
                            start
                            + (samples.shape[0] - 1) * (self.sr / generator.sr)
                            + 1,
                            math.floor((samples.shape[0]) * (self.sr / generator.sr)),
                            endpoint=True,
                        ),
                        np.linspace(
                            math.floor(new_start * self.sr / generator.sr),
                            (new_start + samples.shape[0] - 1) * self.sr / generator.sr,
                            samples.shape[0],
                            endpoint=True,
                        ),
                        samples,
                    )
                    return result
                else:
                    result = np.interp(
                        np.linspace(
                            start,
                            start + sample_count - 1,
                            sample_count,
                            endpoint=True,
                        ),
                        np.linspace(
                            math.floor(new_start * self.sr / generator.sr),
                            (new_start + new_sample_count - 1) * self.sr / generator.sr,
                            samples.shape[0],
                            endpoint=True,
                        ),
                        samples,
                    )
                    return result

        else:
            if convert_num_to_array:
                return np.full((sample_count,), generator, dtype=np.float64)
            else:
                return generator

    def _get_samples_from_multiple(
        self,
        generators: Iterable[GenOrNum],
        sample_count: int,
        start: int,
        channel: int,
        convert_num_to_array: bool = True,
    ) -> tuple[int, tuple[np.ndarray | float | int, ...]]:
        """Get samples from multiple generators and return the minimum length of the samples.

        Parameters
        ----------
        generators
            The generators to get samples from.
        sample_count
            The number of samples to get.
        start
            The starting sample index.
        sr
            The sample rate.
        convert_num_to_array
            Whether to convert numbers to numpy arrays.

        Returns
        -------
        int
            The amount of samples generated.
        tuple
            A tuple of the samples.

        Examples
        --------
        >>> _get_samples_from_multiple([SinOsc(440), 0.5], 10, 0, 44100)
        (10, (np.array([0.0, ...]), np.array([0.5, ...])))

        >>> _get_samples_from_multiple([SinOsc(440), 0.5], 10, 0, 44100, convert_num_to_array=False)
        (10, (np.array([0.0, ...]), 0.5))
        """
        samples = [
            self._get_samples(
                gen,
                sample_count,
                start,
                convert_num_to_array=convert_num_to_array,
                channel=channel,
            )
            for gen in generators
        ]
        min_len = min(
            (len(s) for s in samples if isinstance(s, np.ndarray)), default=sample_count
        )
        if min_len is None:
            return (min_len, tuple(samples))
        else:
            return (
                min_len,
                tuple(s[:min_len] if isinstance(s, np.ndarray) else s for s in samples),
            )

    # endregion

    # region - PUBLIC -
    def reset(self) -> None:
        """Reset the generator to its initial state."""
        self.states = None

    def get_nodes(self) -> dict[str, GenOrNum]:
        """Returns a dictionary of the generators in the nodes mapped to their names.

        Per default, this method returns all generators from self._node_items. So
        if you have items with the semantic meaning of a node, which however are not
        added as nodes to the generator, you might want to override this method.
        """
        return {name: node.gen for name, node in self._node_items.items()}

    def create_graph(self, additional_attr: list[str] = []) -> graphviz.Digraph:
        """Create a graphviz graph of the generator.

        This method traverses the generator nodes recursively using `Agen.get_nodes()`
        to create the graph.

        Parameters
        ----------
        additional_attr
            Additional attributes of the AGens that should be displayed in the graph.

        Returns
        -------
        graphviz.Digraph
            The graph describing the generator.
        """
        try:
            import graphviz
        except Exception as e:
            raise ImportError("AGen.create_graph requires graphviz") from e

        def get_label(agen: AGen) -> str:
            # TODO: The labels are not perfectly centered when rendering in jupyter notebooks
            label = agen.label
            if len(additional_attr) > 0:
                captions = {
                    attr: getattr(agen, attr)
                    for attr in additional_attr
                    if hasattr(agen, attr)
                }

                captions_str = "<BR />".join(
                    f'<FONT POINT-SIZE="10">{k}: {v}</FONT>'
                    for k, v in captions.items()
                )
                label = f"<{agen.label}<BR />{captions_str}>"
            return label

        graph = graphviz.Digraph()
        graph.node(str(self.uuid), label=get_label(self), pos="0,0!")

        finished_nodes: set[uuid.UUID] = set()
        remaining_nodes: deque[AGen] = deque([self])
        while remaining_nodes:
            current = remaining_nodes.popleft()
            finished_nodes.add(current.uuid)
            if not isinstance(current, AGen):
                graph.node(str(uuid.uuid4()), label=str(current))
                continue
            for name, gen in current.get_nodes().items():
                if isinstance(gen, AGen):
                    node_uuid = str(gen.uuid)
                    node_label = get_label(gen)
                    if gen.uuid not in finished_nodes and gen not in remaining_nodes:
                        remaining_nodes.append(gen)
                else:
                    node_uuid = str(uuid.uuid4())
                    node_label = f"{gen:.2f}" if isinstance(gen, float) else str(gen)
                graph.edge(node_uuid, str(current.uuid), label=name)

                graph.node(node_uuid, label=node_label)

        return graph

    def generate(self, sample_count: int, start: int, channel: int) -> np.ndarray:
        """Generate the next `sample_count` samples starting from `start`.

        Parameters
        ----------
        sample_count
            The number of samples to generate.
        start
            The starting sample index.

        Returns
        -------
        np.ndarray
            The generated samples.
        """
        if self.__adaptive_sr and self.sr == -1:
            raise ValueError(
                "Could not determine sample rate as sample rate is set to adaptive but "
                "no nodes were added to the generator. "
                "Please either specify a sampling rate or ensure that the generator "
                "has at least one node that is a generator (and not a number)"
            )
        if self.states is None:
            self.states = [AGenState() for _ in range(self.channels)]
        state = self.states[channel]
        # TODO: These are too many indentations. Refactor this.
        if not state.finished:
            if state.length >= start + sample_count:
                samples = state.get_from_cache(sample_count, start)
            elif start < state.length:
                self.generate(
                    sample_count - (state.length - start),
                    state.length,
                    channel=channel,
                )
                samples = state.get_from_cache(sample_count, start)
            else:
                if state.length < start:
                    # if state.length == 0:
                    #     self.generate(start + sample_count, 0, channel=channel)
                    #     samples = state.get_from_cache(sample_count, start)
                    # else:
                    raise ValueError(
                        "Cannot skip samples while generating: "
                        f"State is currently at sample {state.length}, requested start is {start}"
                    )
                else:
                    self.__prepare_nodes(sample_count, start, channel=channel)
                    self.state = state
                    new_samples = self._generate_new(
                        sample_count=sample_count,
                        start=start,
                        channel=channel,
                    )
                    self.state: AGenState = None  # type: ignore
                    self.__clear_nodes()
                    state.add_to_cache(new_samples)
                    samples = new_samples
        else:
            samples = state.get_from_cache(sample_count, start)

        if samples.shape[0] < sample_count:
            state.mark_finished()
            match self.done:
                case DoneAction.STOP:
                    return samples
                case DoneAction.LAST:
                    return np.concatenate(
                        [
                            samples,
                            np.full(
                                sample_count - samples.shape[0],
                                state.get_from_cache(
                                    1,
                                    state.length - 1,
                                ).item(),
                            ),
                        ]
                    )
                case DoneAction.LOOP:
                    remaining = sample_count - samples.shape[0]
                    looped = state.get_from_cache(
                        remaining, (start + samples.shape[0]) % state.length
                    )
                    if looped.shape[0] < remaining:
                        # If the length of the generator is smaller than the remaining
                        # samples, we need to loop the samples until we have enough.
                        looped = np.concatenate(
                            [
                                looped,
                                np.tile(
                                    # Start from the beginning for the remaining repetitions
                                    state.get_from_cache(state.length, 0),
                                    math.ceil(
                                        (remaining - looped.shape[0]) / state.length
                                    ),
                                ),
                            ]
                        )[:remaining]
                    return np.concatenate([samples, looped])
                case DoneAction.ZERO:
                    return np.concatenate([samples, np.zeros(sample_count - samples.shape[0])])

        return samples

    def __gen_asig_until_done(
        self,
        block_size: int = 10_000,
    ) -> Asig:
        """Generate an Asig until the generator is done.

        This method generates samples in blocks of `block_size` until the generator
        terminates and concats them.

        Note that this method will only terminate if the generator has `DoneAction.STOP`
        as the done action and if the generator will eventually stop generating samples.

        Parameters
        ----------
        start
            The starting sample index.
        block_size
            The block size to use for generating samples.

        Returns
        -------
        Asig
            The generated Asig.
        """
        blocks = []
        current_sample = 0
        while True:
            if (
                config.MAX_GEN_TIME is not None
                and current_sample / self.sr > config.MAX_GEN_TIME
            ):
                warnings.warn("Generated Asig is too long. Stopping generation.")
                break
            sig = [
                self.generate(
                    block_size,
                    channel=channel,
                    start=current_sample,
                )
                for channel in range(self.channels)
            ]
            min_len = min((s.shape[0] for s in sig))
            blocks.append(np.stack([s[:min_len] for s in sig], axis=1))
            current_sample += block_size
            if min_len < block_size:
                break

        return Asig(
            sig=np.concatenate(blocks).squeeze(),
            sr=self.sr,
            label=self.label,
            channels=self.channels,
            cn=self.cn,
        )

    def gen_asig(
        self,
        sample_count: int | None = None,
        seconds: float | None = None,
        block_size: int = 10_000,
    ) -> Asig:
        """Generate an Asig from the generator.

        Parameters
        ----------
        If either `sample_count` or `seconds` is provided, all samples will be generated
        in one large block.

        If neither `sample_count` nor `seconds` is provided, this will generate new
        samples in blocks of `block_size` until the generator is done. This only
        terminates if the top level generator has `DoneAction.STOP` as the done action
        and if the generator will eventually stop generating samples.

        sample_count
            The number of samples to generate.
        seconds
            The number of seconds to generate.
        start
            The starting sample index.
        block_size
            The block size to use for generating samples. This parameter is ignored
            if `sample_count` or `seconds` is provided.
        """

        if seconds is not None or sample_count is not None:
            if seconds is not None:
                count = int(seconds * self.sr)
            else:
                count: int = sample_count  # type: ignore

            # sig = self.generate(count, start=0)
            sig = [
                self.generate(
                    count,
                    start=0,
                    channel=channel,
                )
                for channel in range(self.channels)
            ]
            min_len = min((s.shape[0] for s in sig))
            return Asig(
                sig=(
                    np.stack(
                        [s[:min_len] for s in sig],
                        axis=1,
                    ).squeeze()
                ),
                sr=self.sr,
                label=self.label,
                channels=self.channels,
                cn=self.cn,
            )
        else:
            return self.__gen_asig_until_done(block_size=block_size)

    def with_label(self, label: str) -> AGen:
        """Updates the label of the generator and returns it.

        Note that this method does not create a copy of the generator but instead
        mutates the original generator. This is useful to set the label of AGens that
        are implicitly created by operators.

        Parameters
        ----------
        label
            The new label of the generator.
        """
        self.label = label
        return self
    
    def mul(self, x: GenOrNum = 1.0) -> AGen:
        """Multiply generator with x (GenOrNum), equivalent to (self * x). 
        The functional form can be easier to write and chain with other methods.

        Parameters
        ----------
        x: GenOrNum
            The factor (either value or generator).
        """
        return self * x

    def lvl(self, db: GenOrNum = 0) -> AGen:
        """Level generator by x (GenOrNum) where db is in dB units.
        This multiplies self with pam.db_to_amp(db).
        The functional form can be easier to write and chain with other methods.

        Parameters
        ----------
        db: GenOrNum
            The change in deciBel to be applied to self.
        """
        return self * pam.db_to_amp(db)
    
    def add(self, x: GenOrNum = 0.0) -> AGen:
        """Add x (GenOrNum to generator, equivalent to (self + x). 
        The functional form can be easier to write and chain with other methods.

        Parameters
        ----------
        x: GenOrNum
            The term to be added to self (either value or generator).
        """
        return self + x

    def with_done(self, done: DoneAction | str) -> AGen:
        """Wraps this AGen with another AGen with `done` as done action. """
        return DoneGen(self, done)

    def skip(
        self,
        seconds: float | None = None,
        samples: int | None = None,
    ) -> AGen:
        """Create a generator that skips the first `seconds` or `samples` of the
        generator.
        Equivalent to calling `delay` with a negative value.

        Parameters
        ----------
        Either `samples` or `seconds` must be provided.

        samples
            The number of samples to skip.
        seconds
            The number of seconds to skip.
        """
        assert (samples is not None) != (
            seconds is not None
        ), "Exactly one of `samples` or `seconds` must be provided"

        if seconds is not None:
            samples = int(seconds * self.sr)

        return SkipGen(self, samples)  # type: ignore

    def delay(
        self,
        seconds: float | None = None,
        samples: int | None = None,
        padding: PaddingType = PaddingType.ZERO,
    ) -> AGen:
        """Create a delayed version of the generator.

        Parameters
        ----------
        Either `samples` or `seconds` must be provided.

        samples
            The number of samples to delay by.
        seconds
            The number of seconds to delay by.
        padding
            The padding type to use for the delay.
        """
        assert (samples is not None) != (
            seconds is not None
        ), "Exactly one of `samples` or `seconds` must be provided"

        if seconds is not None:
            samples = int(seconds * self.sr)

        return DelayGen(self, samples, padding=padding)  # type: ignore

    def play(
        self, rate: float = 1.0, server=None, onset=0, channel: int = 0, block=False
    ) -> AGen:
        """Play AGen via Aserver, using Aserver.default (if existing)
        kwargs are propagated to Aserver:play(onset=0, out=0)

        Parameters
        ----------
        rate : float
            Playback rate (Default value = 1) NOT YET IMPLEMENTED
        **kwargs : str
            'server' : Aserver
                Set which server to play. e.g. s = Aserver(); s.boot(); asig.play(server=s)

        Returns
        -------
        _ : AGen
            return self
        """
        import pya.aserver

        if server is None:
            server = pya.aserver.Aserver.default
        if rate == 1 and self.sr == server.sr:
            agen = self
        else:
            agen = ResampleGen(self, sr=server.sr, rate=rate)
        server.play_agen(agen, server=server, onset=onset, out=channel, block=block)
        return self

    # endregion

    # region - CLASSMETHODS -
    @classmethod
    def ar(cls, *args, **kwargs) -> AGen:
        """Alternative constructor using config.AUDIO_RATE rate as the sample rate.
        For more information about the parameters, see the main constructor.
        """
        return cls(*args, sr=config.AUDIO_RATE, **kwargs)

    @classmethod
    def kr(cls, *args, **kwargs) -> AGen:
        """Alternative constructor using config.CONTROL_RATE rate as the sample rate.
        For more information about the parameters, see the main constructor.
        """
        return cls(*args, sr=config.CONTROL_RATE, **kwargs)

    @classmethod
    def p(cls, *args, **kwargs) -> PartialAGen:
        """Partial constructor that allows for a more functional-style construction
        of generators. For more information about the parameters, see the main constructor.

        Note that this only works if the first positional argument of the constructor can
        be an AGen.

        Examples
        --------
        >>> gen = Line(0, 100, 1) | SinOsc.p(phase=0.5) # equivalent to SinOsc(Line(0, 100, 1), phase=0.5)

        Parameters
        ----------
        args
            Additional positional arguments to pass to the constructor of the AGen.
        kwargs
            Additional keyword arguments to pass to the constructor of the AGen.

        Returns
        -------
        PartialAGen
            A partial AGen.
        """

        return PartialAGen(cls, *args, **kwargs)

    # endregion

    def apply(self, fun, *args, **kwargs) -> AGen:
        """apply a function by turning it into an AGen using xgen with self as first argument

        Args:
            fun (Callable): a function to process samples,
            e.g. to be used for nonlinear distortion, etc.

        Returns:
            AGen: the XGen Agen that results from xgen(fun), applied to self with given kwargs
        """
        return xgen(fun)(self, *args, **kwargs)

    # custom functions for AGen - to be extended to many usual suspects from numpy, pyamapping, etc
    def arctan(self, **kwargs):
        return self.apply(np.arctan, **kwargs)

    def sqrt(self, **kwargs):
        return self.apply(np.sqrt, **kwargs)

    def abs(self, **kwargs):
        return self.apply(np.abs, **kwargs)

    def sign(self, **kwargs):
        return self.apply(np.sign, **kwargs)

    def sin(self, **kwargs):
        return self.apply(np.sin, **kwargs)

    def cos(self, **kwargs):
        return self.apply(np.cos, **kwargs)

    def exp(self, **kwargs):
        return self.apply(np.exp, **kwargs)

    def log(self, **kwargs):
        return self.apply(np.log, **kwargs)

    def power(self, *args, **kwargs):
        return self.apply(np.power, *args, **kwargs)

    def linlin(self, *args, **kwargs):
        return self.apply(pam.linlin, *args, **kwargs)

    def midi_to_cps(self, **kwargs):
        return self.apply(pam.midi_to_cps, **kwargs)

    def stereo(self) -> AGen:
        return stereo(self, self)
    
    def dup(self, n: int) -> MultiChannelGen:
        """Duplicate AGen n times to create a MultiChannelGen.
        This is inspired from SuperColliders dup() function

        Parameters
        ----------
        n: int
            The number of duplications
        """
        return MultiChannelGen([self]*n)


    def mix(self) -> MixGen:
        """Mix the channels of the generator."""
        return MixGen(self)

    def limit(
        self,
        seconds: int | float | None = None,
        samples: int | None = None,
        done: DoneAction = DoneAction.STOP,
    ) -> TimeLimitGen:
        """Limit the duration of the generator to a certain number of samples or
        seconds.

        Parameters
        ----------
        Either `seconds` or `samples` must be provided.

        seconds
            The number of seconds to limit the generator to.
        samples
            The number of samples to limit the generator.
        done
            The done action to use for the resulting `TimeLimitGen`.

        Returns
        -------
        LimitGen
            A generator that limits the duration of the original
        """
        return TimeLimitGen(self, seconds=seconds, samples=samples, done=done)

    def fade_in(self, seconds: float, curve: float = 1) -> AGen:
        """Wrap the generator in a fade in generator."""
        return FadeInGen(self, seconds, curve)

    def fade_out(self, seconds: float, curve: float = 1) -> AGen:
        """Wrap the generator in a fade out generator."""
        return FadeOutGen(self, seconds, curve)

    def fade(
        self,
        both: float | None = None,
        *,
        in_secs: float | None = None,
        out_secs: float | None = None,
        curve: float = 1,
    ) -> AGen:
        """Wrap the generator in a fade in and fade out generator.

        Parameters
        ----------
        Either `both` or `in_secs` and `out_secs` must be provided.

        both
            The duration of the fade in and fade out.
        in_secs
            The duration of the fade in.
        out_secs
            The duration of the fade out.
        curve
            The curve of the fade.

        """
        assert (both is None) != (
            in_secs is None and out_secs is None
        ), "Exactly one of `both` or `in_secs` and `out_secs` must be provided"

        if both is not None:
            in_secs = out_secs = both

        return FadeInGen(FadeOutGen(self, out_secs, curve), in_secs, curve)  # type: ignore

    def to_sr(self, sr: int = 44100, rate: float = 1.0):
        """resample shortcut - wraps AGen.Resample() around self

        Args:
            sr (float, optional): target sampling rate. Defaults to 44100.
            rate (float, optional): additional resampling rate. Defaults to 1.0.

        Returns:
            AGen: new generator running at target sr
        """
        return ResampleGen(self, rate=rate, sr=sr)




class PartialAGen:
    """Partial constructor for an AGen."""

    def __init__(self, gen_class: type[AGen], *args, **kwargs) -> None:
        self.gen_class = gen_class
        self.args = args
        self.kwargs = kwargs

    def __ror__(self, value: GenOrNum) -> AGen:
        return self.gen_class(value, *self.args, **self.kwargs)  # type: ignore


class SingleChannelGen(AGen):
    """Abstract base class for generators known to have only one channel."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, channels=1, **kwargs)

    @abstractmethod
    def _generate_single(self, sample_count: int, start: int) -> np.ndarray: ...

    def _generate_new(self, sample_count: int, start: int, channel: int) -> np.ndarray:
        return self._generate_single(sample_count=sample_count, start=start)

class DoneGen(AGen):
    def __init__(self, gen: GenOrNum, done: DoneAction | str, *args, **kwargs):
        super().__init__(*args, done=done, **kwargs)
        self._add_node(gen, "gen", convert_num_to_arr=True)

    def get_nodes(self) -> dict[str, AGen | Any]:
        return {
            "done": self.done, 
            **super().get_nodes()
        }
    
    def _generate_new(self, sample_count: int, start: int, channel: int) -> np.ndarray:
        return self.nodes["gen"]  # type: ignore

    

class MultiChannelGen(AGen):
    def __init__(
        self,
        gens: Sequence[GenOrNum],
        cn: list[str] | None = None,
        *args,
        **kwargs,
    ) -> None:
        self._gens = gens
        self._cn = cn
        self.gen_channels = []
        channel = 0
        for gen in gens:
            num_channels = gen.channels if isinstance(gen, AGen) else 1
            for i in range(num_channels):
                self.gen_channels.append((gen, i))
                channel += 1
        super().__init__(*args, channels=channel, sr=get_max_sr(gens), cn=cn, **kwargs)

    def get_nodes(self):
        return {f"Channel {i}": gen for i, gen in enumerate(self._gens)}

    def _generate_new(self, sample_count: int, start: int, channel: int) -> np.ndarray:
        gen, i = self.gen_channels[channel]
        return self._get_samples(
            gen,
            sample_count,
            start,
            channel=i,
            convert_num_to_array=True,
        )  # type: ignore


class ChannelSelectorGen(AGen):
    def __init__(
        self,
        gen: AGen,
        index: int | slice | list[bool] | list[int],
        *args,
        **kwargs,
    ) -> None:
        self._index = index
        self._gen = gen
        if isinstance(index, int):
            channels = 1
            self._mapping = [index]
        elif isinstance(index, str):
            assert self._gen.cn is not None, "No channel names defined"
            channels = 1
            self._mapping = [self._gen.cn.index(index)]
        elif isinstance(index, list):
            assert len(index) > 0, "Index must not be empty."
            if isinstance(index[0], bool):
                assert (
                    len(index) == gen.channels
                ), "Length of index must match number of channels."
                channels = sum(index)
                self._mapping = [i for i, val in enumerate(index) if val]
            elif isinstance(index[0], str):
                assert self._gen.cn is not None, "No channel names defined"
                self._mapping = [self._gen.cn.index(n) for n in index]
                channels = len(index)
            else:
                assert isinstance(
                    index[0], int
                ), "Index must be a list of integers or bools."
                channels = len(index)
                self._mapping = index
        else:
            start, stop, step = index.indices(gen.channels)
            channels = len(range(start, stop, step))
            self._mapping = list(range(start, stop, step))
        super().__init__(*args, channels=channels, sr=gen.sr, **kwargs)

    def get_nodes(self) -> dict[str, AGen | Any]:
        return {"gen": self._gen, "index": self._index}

    def __get_channel(self, channel: int) -> int:
        return self._mapping[channel]

    def _generate_new(self, sample_count: int, start: int, channel: int) -> np.ndarray:
        channel = self.__get_channel(channel)
        return self._gen.generate(
            sample_count=sample_count,
            channel=channel,
            start=start,
        )


def multi_channel(*gens: GenOrNum, cn: list[str] | None = None) -> MultiChannelGen:
    return MultiChannelGen(list(gens), cn=cn)


def stereo(left: GenOrNum, right: GenOrNum) -> MultiChannelGen:
    return MultiChannelGen([left, right], cn=["l", "r"])


class TimeLimitGen(AGen):
    def __init__(
        self,
        gen: AGen,
        seconds: int | float | None = None,
        samples: int | None = None,
        *args,
        sr: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, sr=sr, **kwargs)
        self._gen = gen
        if seconds is not None:
            samples = int(seconds * gen.sr)
        if samples is None:
            raise ValueError("Either `seconds` or `samples` must be provided.")
        self._samples = samples
        self._add_node(gen, "gen", convert_num_to_arr=True)

    def get_nodes(self):
        return {"limit": self._samples / self.sr, **super().get_nodes()}

    def _generate_new(self, sample_count, start, channel):
        if start + sample_count < self._samples:
            return self.nodes["gen"]
        return self.nodes["gen"][: self._samples - start]  # type: ignore


class DelayGen(AGen):
    """Generator for delaying another generator.

    Parameters
    ----------
    gen
        The generator to delay.
    samples
        The delay in samples.
    padding
        The padding type to use for the delay.
    """

    def __init__(
        self,
        gen: AGen,
        samples: int,
        *args,
        padding: PaddingType = PaddingType.ZERO,
        sr: int | None = None,
        **kwargs,
    ) -> None:
        assert samples >= 0, "samples must be greater or equal to 0."
        self._gen = gen
        self.delay_samples = samples
        self.padding = padding
        if sr is None:
            sr = gen.sr
        super().__init__(
            sr=sr,
            channels=gen.channels,
            cn=gen.cn,
            *args,
            **kwargs,
        )

    def get_nodes(self) -> dict[str, AGen | float | int]:
        return {
            "gen": self._gen,
            "delay": self.delay_samples,
        }

    def _generate_new(self, sample_count: int, start: int, channel: int) -> np.ndarray:
        if start > self.delay_samples:
            return self._get_samples(
                self._gen, sample_count, start - self.delay_samples, channel=channel, convert_num_to_array=True,
            )  # type: ignore
        else:
            padding_value = self.state.data.get("padding_value", None)
            if padding_value is None:
                match self.padding:
                    case PaddingType.FIRST:
                        padding_value = self._get_samples(
                            self._gen,
                            sample_count=1,
                            start=0,
                            channel=channel,
                            convert_num_to_array=True,
                        ).item()  # type: ignore
                    case PaddingType.ZERO:
                        padding_value = 0
                    case _:
                        raise ValueError(
                            f"Invalid padding type: {self.padding}. Must be one of: {', '.join(PaddingType)}"
                        )
                self.state.data["padding_value"] = padding_value
            gen_start = min(self.delay_samples - start, sample_count)
            return np.concatenate(
                [
                    padding_value * np.ones(gen_start),
                    self._gen.generate(
                        sample_count=sample_count - gen_start,
                        start=0,
                        channel=channel,
                    ),
                ]
            )


class SkipGen(AGen):
    """Generator for skipping the first `samples` of another generator.

    Parameters
    ----------
    gen
        The generator to skip.
    samples
        The number of samples to skip.
    """

    def __init__(self, gen: AGen, samples: int, sr: int | None = None, *args, **kwargs):
        if sr is None:
            sr = gen.sr
        super().__init__(
            *args,
            sr=sr,
            channels=gen.channels,
            **kwargs,
        )
        assert samples >= 0, "Samples must be greater or equal to 0."
        self._gen = gen
        self._samples = samples

    def get_nodes(self):
        return {"gen": self._gen, "samples": self._samples, **super().get_nodes()}

    def _generate_new(self, sample_count, start, channel):
        if start == 0:
            self._get_samples(self._gen, self._samples, 0, channel)
        return self._gen.generate(sample_count, start + self._samples, channel)


def get_max_sr(gens: Iterable[GenOrNum]) -> int:
    return max(
        [gen.sr for gen in gens if isinstance(gen, AGen)],
        default=config.AUDIO_RATE,
    )


def get_max_channel(gens: Iterable[GenOrNum]) -> int:
    max_channels = 1
    for gen in gens:
        if isinstance(gen, AGen) and gen.channels > 1:
            if max_channels > 1 and gen.channels != max_channels:
                raise ValueError(
                    "Cannot combine AGen with m channels with AGen with n channels "
                    "where m != n; n, m > 1."
                )
            max_channels = gen.channels
    return max_channels


class ConcatGen(AGen):
    def __init__(self, *gens: GenOrNum, **kwargs):
        super().__init__(
            sr=get_max_sr(gens),
            label="Concat",
            channels=get_max_channel(gens),
            **kwargs,
        )
        self._gens = gens

    def _generate_new(self, sample_count: int, channel: int, **_) -> np.ndarray:
        current_gen = self.state.data.get("current_gen", 0)
        gen_sample = self.state.data.get("gen_sample", 0)
        samples = []
        count = 0
        # Generate samples from the current generator and continue to the next one
        # if necessary until we have enough samples or we run out of generators
        while count < sample_count:
            s: np.ndarray = self._get_samples(
                self._gens[current_gen],
                sample_count - count,
                start=gen_sample,
                channel=channel,
                convert_num_to_array=True,
            )  # type: ignore
            samples.append(s)
            count += s.shape[0]
            gen_sample += s.shape[0]
            if s.shape[0] < sample_count - count:
                current_gen += 1
                gen_sample = 0
                if current_gen >= len(self._gens):
                    break
        self.state.data["current_gen"] = current_gen
        self.state.data["gen_sample"] = gen_sample
        return np.concatenate(samples)


class AddGen(AGen):
    """Generator for addition of multiple generators.
    At least one of the generators must be an AGen.

    Parameters
    ----------
    gens
        The generators to add.
    """

    def __init__(self, *gens: GenOrNum, **kwargs):
        super().__init__(sr=None, label="+", **kwargs)
        for i, gen in enumerate(gens):
            self._add_node(gen, f"s_{i}", convert_num_to_arr=True)

    def _generate_new(self, **_) -> np.ndarray:
        return np.sum(list(self.nodes.values()), axis=0)  # type: ignore


class MulGen(AGen):
    """Generator for multiplication of multiple generators.
    At least one of the generators must be an AGen.

    Parameters
    ----------
    gens
        The generators to multiply.
    """

    def __init__(self, *gens: GenOrNum, **kwargs):
        super().__init__(sr=None, label="*", **kwargs)
        for i, gen in enumerate(gens):
            self._add_node(gen, f"f_{i}", convert_num_to_arr=True)

    def _generate_new(self, sample_count: int, start: int, channel: int) -> np.ndarray:
        return np.prod(list(self.nodes.values()), axis=0)  # type: ignore


class PowGen(AGen):
    """Generator for exponentiation of two generators.
    At least one of gen and exp must be an AGen.

    Parameters
    ----------
    gen
        The base generator
    exp
        The exponent generator
    """

    def __init__(self, base: GenOrNum, exp: GenOrNum, *args, **kwargs):
        super().__init__(*args, sr=None, label="^", **kwargs)

        self._add_node(base, "base")
        self._add_node(exp, "exp")

    def _generate_new(self, sample_count: int, start: int, channel: int) -> np.ndarray:
        return self.nodes["base"] ** self.nodes["exp"]  # type: ignore


class DivGen(AGen):
    """Generator for division of two generators.
    At least one of gen1 and gen2 must be an AGen.

    Parameters
    ----------
    dividend
        The dividend generator
    divisor
        The divisor generator
    """

    def __init__(self, dividend: GenOrNum, divisor: GenOrNum, *args, **kwargs):
        super().__init__(*args, sr=None, label="/", **kwargs)

        self._add_node(dividend, "dividend")
        self._add_node(divisor, "divisor")

    def _generate_new(self, sample_count: int, start: int, channel: int) -> np.ndarray:
        return self.nodes["dividend"] / self.nodes["divisor"]  # type: ignore


def xgen(
    fun: Callable[..., np.ndarray],
    *gen_args,
    label: str | None = None,
    **gen_kwargs,
) -> Callable[..., AGen]:
    if label is None:
        label = fun.__name__

    if "sr" not in gen_kwargs:
        # If the sample rate is not specified, make it dynamic
        gen_kwargs["sr"] = None

    class XGen(SingleChannelGen):
        def __init__(self, *args, **kwargs):
            self._kwargs = kwargs
            self._args = args
            super().__init__(*gen_args, label=label, **gen_kwargs)
            for i, arg in enumerate(args):
                self._add_node(arg, f"arg.{i}")

            for key, value in kwargs.items():
                self._add_node(value, f"kwarg.{key}")

        def _generate_single(self, sample_count: int, start: int) -> np.ndarray:
            generated_args = [self.nodes[f"arg.{i}"] for i in range(len(self._args))]
            generated_kwargs = {key: self.nodes[f"kwarg.{key}"] for key in self._kwargs}
            return fun(*generated_args, **generated_kwargs)

    return XGen


class FadeInGen(SingleChannelGen):
    """Generator for fading in another generator.

    Parameters
    ----------
    gen
        The generator to fade in.
    duration
        The duration of the fade in in seconds.
    curve
        The curve of the fade in.
    """

    def __init__(self, gen: GenOrNum, duration: float, curve: int | float = 1, *args, **kwargs):
        super().__init__(*args, sr=None, **kwargs)
        self._add_node(gen, "gen", convert_num_to_arr=True)
        self.duration = duration
        self.curve = curve

    def get_nodes(self):
        return {"duration": self.duration, "curve": self.curve, **super().get_nodes()}

    def _generate_single(self, sample_count, start):
        samples: np.ndarray = self.nodes["gen"]  # type: ignore

        end = self.duration * self.sr

        if start >= end:
            return samples

        start_time = start / self.sr
        end_sample = math.floor(min((start + sample_count), self.duration * self.sr))
        slope = 1 / self.duration
        mul = (
            np.linspace(
                start=slope * start_time,
                stop=slope * end_sample / self.sr,
                num=min(sample_count, max(end_sample - start, 0)),
                endpoint=False,
            )
            ** self.curve
        )
        samples = samples.copy()
        samples[: mul.shape[0]] *= mul

        return samples


class FadeOutGen(AGen):
    """Generator for fading out another generator.

    Note that this generator needs to precompute `duration * sr + block_size` samples of
    the generator to start the fade out at the correct time.

    Parameters
    ----------
    gen
        The generator to fade out.
    duration
        The duration of the fade out in seconds.
    curve
        The curve of the fade out.
    """

    def __init__(self, gen: GenOrNum, duration: float, curve: int | float = 1, *args, **kwargs):
        super().__init__(*args, sr=None, **kwargs)
        self._add_node(gen, "gen", convert_num_to_arr=True)
        self.gen = gen
        self.duration = duration
        self.curve = curve

    def get_nodes(self):
        return {"duration": self.duration, "curve": self.curve, **super().get_nodes()}

    def _generate_new(self, sample_count, start, channel):
        start_fade = self.state.data.get("start_fade", None)
        if start_fade is None:
            look_ahead_len = math.ceil(self.sr * self.duration) + sample_count
            look_ahead: np.ndarray = self._get_samples(
                self.gen,
                look_ahead_len,
                start,
                channel=channel,
                convert_num_to_array=True,
            )  # type: ignore
            if look_ahead.shape[0] < look_ahead_len:
                if look_ahead.shape[0] < math.ceil(self.sr * self.duration):
                    warnings.warn(
                        "AGen too short for fade_out - adapting fade_out time"
                    )
                    self.duration = look_ahead.shape[0] / self.sr
                start_fade = (
                    start + look_ahead.shape[0] - math.ceil(self.sr * self.duration)
                )
                self.state.data["start_fade"] = start_fade

        samples: np.ndarray = self.nodes["gen"]  # type: ignore
        if start_fade is None or start + sample_count <= start_fade:
            return samples

        new_start = max(start, start_fade)
        start_time = (new_start - start_fade) / self.sr
        new_sample_count = sample_count - max(start_fade - start, 0)
        end_sample = math.floor(
            min((new_start + new_sample_count - start_fade), self.sr * self.duration)
        )
        slope = -1 / self.duration
        mul = (
            np.linspace(
                start=1 + slope * start_time,
                stop=1 + slope * end_sample / self.sr,
                num=min(new_sample_count, end_sample, samples.shape[0]),
                endpoint=False,
            )
            ** self.curve
        )

        if mul.shape[0] > 0:
            # Copy the samples to avoid modifying the original array
            samples = samples.copy()
            samples[-mul.shape[0] :] *= mul

        return samples


class MixGen(AGen):
    """Mixes the channels of a generator into a single channel.

    Parameters
    ----------
    gen
        The generator to mix.
    """

    def __init__(self, gen: AGen, *args, **kwargs):
        super().__init__(sr=gen.sr, channels=1, *args, **kwargs)
        self.gen = gen

    def _generate_new(self, sample_count, start, channel):
        sigs: list[np.ndarray] = [
            self._get_samples(
                self.gen,
                sample_count,
                start,
                channel=i,
                convert_num_to_array=True,
            )
            for i in range(self.gen.channels)
        ]  # type: ignore
        min_len = min((s.shape[0] for s in sigs))
        return np.sum([s[:min_len] for s in sigs], axis=0)

class ResampleGen(AGen):
    """Resample an AGen to sr apply resampling rate

    Parameters
    ----------
    gen (SingleChannelGen)
        The AGen that should be resampled
    rate (float)
        The resampling rate (default 1.0), e.g. 0.5 stretches sound by factor 2
    sr (integer)
        the target rate (as default kwarg of the AGen)        
    """
    def __init__(self, gen: AGen, rate: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = AGenState()
        self.sample_incr = gen.sr / self.sr * rate
        self.agen = gen
        self.rate = rate
        self.channels = gen.channels

    def _generate_new(
        self,
        sample_count: int,  # The amount of samples that should be generated
        start: int,  # The index of the first sample
        channel: int,
    ) -> np.ndarray:
        # TH TODO: make rate an GenOrNum, check multi-channel
        gen_pos = self.state.data.get("gen_pos", 0)
        start_idx = max(0, int(gen_pos)-1)
        gen_stop_pos = gen_pos + self.sample_incr * sample_count
        n_render = int(gen_stop_pos + 1) - start_idx + 1
        src_sig = self.agen.generate(n_render, start_idx, channel=channel)
        n_pts = src_sig.shape[0]
        src_pos = np.arange(0, n_pts) + start_idx # faster than np.linspace

        if n_pts < n_render: # limit output if input end was reached 
            stop_index = start_idx + n_pts
            m = int((min(stop_index, gen_stop_pos) - gen_pos) / self.sample_incr)
            dest_max_pos = gen_pos + m * self.sample_incr
        else:
            m = sample_count
            dest_max_pos = gen_stop_pos
        dest_pos = np.linspace(gen_pos, dest_max_pos, m, endpoint=False)
        dest_sig = np.interp(dest_pos, src_pos, src_sig)

        self.state.data["gen_pos"] = gen_pos + m * self.sample_incr

        if n_pts > 0:
            return dest_sig
        else:
            return np.empty(0)
