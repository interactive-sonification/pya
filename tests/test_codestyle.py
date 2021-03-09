import unittest
import pycodestyle


class TestCodeFormat(unittest.TestCase):

    def test_conformance(self):
        """Test that we conform to PEP-8."""
        # E731 ignores lamda, W291 trailing whitespace
        # W391 blank line at end of file
        # W292 no newline at end of file
        # E722 bare except
        style = pycodestyle.StyleGuide(quiet=False,
                                       ignore=['E501', 'E731', 'W291', 'W504',
                                               'W391', 'W292', 'E722', 'E402'])
        # style.input_dir('../../pya')
        style.input_dir('./pya')
        style.input_dir('./tests')
        result = style.check_files()
        self.assertEqual(0, result.total_errors,
                         "Found code style errors (and warnings).")
