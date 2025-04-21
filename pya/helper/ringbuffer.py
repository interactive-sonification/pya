import numpy as np


class RingBuffer:
    def __init__(self, frames: int, channels: int, dtype: np.dtype = np.float32):
        if not (frames & (frames - 1) == 0):
            raise ValueError("frames must be a power of 2")
        self._frames = frames
        self._channels = channels
        self._dtype = dtype
        # bitmask for the circular buffer, faster than modulo
        self._bitmask = frames - 1
        self._buffer = np.zeros((frames, channels), dtype=dtype)
        self.write_index = 0
        self.read_index = 0

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def buffer(self) -> np.ndarray:
        return self._buffer

    def write(self, data: np.ndarray) -> int:
        """Write data to the ring buffer, returns the number of frames written."""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        if data.shape[0] > self.frames:
            raise ValueError(
                "data frames size must not be greater than the ring buffer frames size"
            )

        if data.shape[1] > self.channels:
            raise ValueError(
                "data channels size must not be greater than the ring buffer channels size"
            )

        frames_to_write = data.shape[0]
        channels_to_write = data.shape[1]
        # Handle potential wrap-around
        first_chunk = min(frames_to_write, self.frames - self.write_index)
        self.buffer[
            self.write_index: self.write_index + first_chunk, :channels_to_write
        ] = data[:first_chunk, :channels_to_write]

        if first_chunk < frames_to_write:
            second_chunk = frames_to_write - first_chunk
            self.buffer[:second_chunk, :channels_to_write] = data[
                first_chunk:frames_to_write, :channels_to_write
            ]

        self.write_index = (self.write_index + frames_to_write) & self._bitmask
        return frames_to_write

    def read(self, frames: int) -> np.ndarray:
        """Read data from the ring buffer, returns the number of frames read."""
        if frames > self.frames:
            raise ValueError(
                "frames must not be greater than the ring buffer frames size"
            )

        frames_available = self.available_read()
        frames_to_read = min(frames, frames_available)

        result = np.zeros((frames_to_read, self.channels), dtype=self.dtype)

        if frames_to_read == 0:
            return result

        first_chunk = min(frames_to_read, self.capacity - self.read_pos)
        result[:first_chunk] = self.buffer[self.read_pos: self.read_pos + first_chunk]

        # If we need to wrap around
        if first_chunk < frames_to_read:
            second_chunk = frames_to_read - first_chunk
            result[first_chunk:] = self.buffer[:second_chunk]

        if advance:
            self.read_pos = (self.read_pos + frames_to_read) % self.capacity

        return result

    def available_read(self) -> int:
        if self.write_index >= self.read_index:
            return self.write_index - self.read_index
        else:
            return self.frames - self.read_index + self.write_index

    def available_write(self) -> int:
        return self.frames - self.available_read() - 1

    def clear(self):
        self.write_index = 0
        self.read_index = 0
        self._buffer.fill(0)
