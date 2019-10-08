from pya import Asig
from unittest import TestCase


class TestLoadFile(TestCase):
    """Test loading audio file."""
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_wav(self):
        asig = Asig("./examples/samples/stereoTest.wav")
        self.assertEqual(2, asig.channels)

    def test_aiff(self):
        asig = Asig("./examples/samples/notes_sr32000_stereo.aif")
        self.assertEqual(32000, asig.sr)

    def test_mp3(self):
        asig = Asig("./examples/samples/ping.mp3")
        self.assertEqual(34158, asig.samples)