import unittest

from pya.agen.lib import (BLIT, BPF, BRF, HPF, LPF, BLPulse, BLSaw, LFPulse,
                          LFSaw, Line, OneZero, Pan2, PanAz, SeqAGen, SinOsc)


class GenOrNumArgsTest(unittest.TestCase):
    """Tests filters implemented with numba with numbers and AGens as arguments to ensure
    that the type signatures are correct"""

    def test_bpf(self):
        BPF(SinOsc(freq=10), f_0=100, bw=3).gen_asig(seconds=1)
        BPF(SinOsc(freq=10), f_0=100.0, bw=3.0).gen_asig(seconds=1)
        BPF(SinOsc(freq=10), f_0=Line(1, 100, 1), bw=Line(1, 10, 1)).gen_asig(seconds=1)

    def test_brf(self):
        BRF(SinOsc(freq=10), f_0=100, bw=3).gen_asig(seconds=1)
        BRF(SinOsc(freq=10), f_0=100.0, bw=3.0).gen_asig(seconds=1)
        BRF(SinOsc(freq=10), f_0=Line(1, 100, 1), bw=Line(1, 10, 1)).gen_asig(seconds=1)

    def test_hpf(self):
        HPF(SinOsc(freq=10), freq=100).gen_asig(seconds=1)
        HPF(SinOsc(freq=10), freq=100.0).gen_asig(seconds=1)
        HPF(SinOsc(freq=10), freq=Line(1, 100, 1)).gen_asig(seconds=1)

    def test_lpf(self):
        LPF(SinOsc(freq=10), freq=100).gen_asig(seconds=1)
        LPF(SinOsc(freq=10), freq=100.0).gen_asig(seconds=1)
        LPF(SinOsc(freq=10), freq=Line(1, 100, 1)).gen_asig(seconds=1)

    def test_one_zero(self):
        OneZero(SinOsc(freq=10), coef=0.5).gen_asig(seconds=1)
        OneZero(SinOsc(freq=10), coef=0).gen_asig(seconds=1)
        OneZero(SinOsc(freq=10), coef=Line(0, 1, 1)).gen_asig(seconds=1)

    def test_sinosc(self):
        SinOsc(freq=10).gen_asig(seconds=1)
        SinOsc(freq=10.0).gen_asig(seconds=1)
        SinOsc(freq=Line(1, 100, 1)).gen_asig(seconds=1)

    def test_lfpulse(self):
        LFPulse(freq=10, width=1, phase=1).gen_asig(seconds=1)
        LFPulse(freq=10.0, width=0.5, phase=1.0).gen_asig(seconds=1)
        LFPulse(
            freq=Line(1, 100, 1), width=Line(0, 1, 1), phase=Line(0, 1, 1)
        ).gen_asig(seconds=1)

    def test_lfsaw(self):
        LFSaw(freq=10, phase=1).gen_asig(seconds=1)
        LFSaw(freq=10.0, phase=1.0).gen_asig(seconds=1)
        LFSaw(freq=Line(1, 100, 1), phase=Line(0, 1, 1)).gen_asig(seconds=1)

    def test_blpulse(self):
        BLPulse(freq=10, width=1).gen_asig(seconds=1)
        BLPulse(freq=10.0, width=0.5).gen_asig(seconds=1)
        BLPulse(freq=Line(1, 100, 1), width=Line(0, 1, 1)).gen_asig(seconds=1)

    def test_blsaw(self):
        BLSaw(freq=10).gen_asig(seconds=1)
        BLSaw(freq=10.0).gen_asig(seconds=1)
        BLSaw(freq=Line(1, 100, 1)).gen_asig(seconds=1)

    def test_blit(self):
        BLIT(freq=10).gen_asig(seconds=1)
        BLIT(freq=10.0).gen_asig(seconds=1)
        BLIT(freq=Line(1, 100, 1)).gen_asig(seconds=1)

    def test_pan2(self):
        Pan2(SinOsc(freq=10), pos=0).gen_asig(seconds=1)
        Pan2(SinOsc(freq=10), pos=0.5).gen_asig(seconds=1)
        Pan2(SinOsc(freq=10), pos=Line(-1, 1, 1)).gen_asig(seconds=1)

    def test_panaz(self):
        PanAz(4, 1, 0, 2, 0).gen_asig(seconds=1)
        PanAz(4, 1.0, 0.0, 2.0, 0.0).gen_asig(seconds=1)
        PanAz(
            4, SinOsc(freq=10), Line(-1, 1, 1), Line(0.1, 3, 1), Line(0, 1, 1)
        ).gen_asig(seconds=1)

    def test_seq_agen(self):
        SeqAGen([(0.0, 1)]).gen_asig(seconds=1)
        SeqAGen([(0.0, 1.0)]).gen_asig(seconds=1)
        SeqAGen([(0.0, Line(0, 1, 1))]).gen_asig(seconds=1)


if __name__ == "__main__":
    unittest.main()
