import unittest
import pycodestyle


class TestCodeFormat(unittest.TestCase):

    def test_conformance(self):
        """Test that we conform to PEP-8."""
        # E731 ignores lamda, W291 trailing whitespace
        style = pycodestyle.StyleGuide(quiet=False, ignore=['E501', 'E731', 'W291'])
        style.input_dir('../../pya')
        # style.input_dir('tests')
        result = style.check_files()
        self.assertEqual(0, result.total_errors,
                         "Found code style errors (and warnings).")
