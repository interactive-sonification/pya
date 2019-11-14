from abc import abstractmethod, ABC


class BackendBase(ABC):

    @abstractmethod
    def get_device_count(self):
        raise NotImplementedError

    @abstractmethod
    def get_device_info_by_index(self):
        raise NotImplementedError

    @abstractmethod
    def get_default_output_device_info(self):
        raise NotImplementedError

    @abstractmethod
    def get_default_input_device_info(self):
        raise NotImplementedError

    @abstractmethod
    def open(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def terminate(self):
        raise NotImplementedError

    @abstractmethod
    def process_buffer(self, *args, **kwargs):
        raise NotImplementedError


class StreamBase(ABC):

    @abstractmethod
    def is_active(self):
        raise NotImplementedError

    @abstractmethod
    def start_stream(self):
        raise NotImplementedError

    @abstractmethod
    def stop_stream(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
