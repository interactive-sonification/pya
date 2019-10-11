from abc import abstractmethod, ABC


class BackendBase(ABC):

    bs = 1024

    @abstractmethod
    def get_device_count(self):
        pass

    @abstractmethod
    def get_device_info_by_index(self):
        pass

    @abstractmethod
    def get_default_output_device_info(self):
        pass

    @abstractmethod
    def get_default_input_device_info(self):
        pass

    @abstractmethod
    def open(self, *args, **kwargs):
        pass

    @abstractmethod
    def terminate(self):
        pass

    @abstractmethod
    def process_buffer(self, *args, **kwargs):
        pass


class StreamBase(ABC):

    @abstractmethod
    def is_active(self):
        pass

    @abstractmethod
    def start_stream(self):
        pass

    @abstractmethod
    def stop_stream(self):
        pass

    @abstractmethod
    def close(self):
        pass
