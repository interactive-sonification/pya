from pya.agen.core import AGen
from pya.agen.types import GenOrNum


class IdentityGen(AGen):
    def __init__(self, gen: GenOrNum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_node(gen, "gen", convert_num_to_arr=True)

    def _generate_new(self, sample_count, start, channel):
        return self.nodes["gen"]
