import os
from unittest import TestCase

import numpy as np

from pya.helper.codec import (
    FFmpegAudioFile,
    SoundFileAudioFile,
    UnsupportedError,
    audio_read,
)


class TestCodec(TestCase):
    """Test codec as well as file loader."""

    def setUp(self):
        self.test_files = {
            "wav": "./examples/samples/stereoTest.wav",
            "aiff": "./examples/samples/notes_sr32000_stereo.aif",
            "mp3": "./examples/samples/ping.mp3",
        }

    def test_wav_soundfile(self):
        """Test WAV file loading with SoundFile backend"""
        with audio_read(self.test_files["wav"]) as audio:
            self.assertIsInstance(audio, SoundFileAudioFile)
            self.assertEqual(2, audio.channels)
            self.assertEqual(44100, audio.samplerate)

            # Test reading data
            data = next(audio.read_data())
            self.assertEqual(np.float32, data.dtype)
            self.assertEqual(2, len(data.shape))  # (samples, channels)

    def test_aiff_soundfile(self):
        """Test AIFF file loading with SoundFile backend"""
        with audio_read(self.test_files["aiff"]) as audio:
            self.assertIsInstance(audio, SoundFileAudioFile)
            self.assertEqual(2, audio.channels)
            self.assertEqual(32000, audio.samplerate)

            # Test reading data
            data = next(audio.read_data())
            self.assertEqual(np.float32, data.dtype)

    def test_mp3_ffmpeg(self):
        """Test MP3 file loading with FFmpeg backend"""
        with audio_read(self.test_files["mp3"]) as audio:
            self.assertIsInstance(audio, FFmpegAudioFile)
            self.assertEqual(1, audio.channels)  # mono MP3
            self.assertEqual(44100, audio.samplerate)

            # Test reading data
            data = next(audio.read_data())
            self.assertEqual(np.float32, data.dtype)

    # TODO: Add flac test
    # def test_flac_soundfile(self):
    #     """Test FLAC file loading with SoundFile backend"""
    #     with audio_read(self.test_files['flac']) as audio:
    #         self.assertIsInstance(audio, SoundFileAudioFile)
    #         self.assertEqual(96000, audio.samplerate)  # 96kHz sample rate

    #         # Test reading data
    #         data = next(audio.read_data())
    #         self.assertEqual(np.float32, data.dtype)

    def test_invalid_file(self):
        """Test handling of invalid/nonexistent files"""
        with self.assertRaises(UnsupportedError):
            audio_read("nonexistent.wav")

    def test_unsupported_format(self):
        """Test handling of unsupported file formats"""
        with self.assertRaises(UnsupportedError):
            audio_read("test.xyz")

    def test_data_reading(self):
        """Test complete data reading process"""
        with audio_read(self.test_files["wav"]) as audio:
            all_data = []
            for block in audio.read_data():
                all_data.append(block)
            data = np.vstack(all_data)

            # Check data properties
            self.assertEqual(np.float32, data.dtype)
            self.assertEqual(2, len(data.shape))
            self.assertEqual(2, data.shape[1])  # stereo

    # TODO: Add multichannel test
    # def test_multichannel(self):
    #     """Test handling of multichannel audio"""
    #     # You'll need a multichannel test file
    #     pass

    # TODO: Add a flac file for testing
    # def test_high_resolution(self):
    #     """Test handling of high resolution audio (24-bit, high sample rate)"""
    #     with audio_read(self.test_files['flac']) as audio:# Assuming 24-bit/96kHz FLAC
    #         data = next(audio.read_data())
    #         self.assertTrue(np.max(np.abs(data)) <= 1.0)  # Check normalization
    #         self.assertEqual(np.float32, data.dtype)

    def test_error_handling(self):
        """Test various error conditions"""
        # Test corrupted file
        with self.assertRaises(UnsupportedError):
            with open("corrupt.wav", "wb") as f:
                f.write(b"NOT_A_WAV_FILE")
            audio_read("corrupt.wav")

        # Clean up
        if os.path.exists("corrupt.wav"):
            os.remove("corrupt.wav")
