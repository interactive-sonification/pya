from setuptools import setup
import os
import codecs
from os.path import join

project_root = os.path.dirname(os.path.abspath(__file__))

version = {}
with open(join(project_root, 'pya/version.py')) as read_file:
    exec(read_file.read(), version)

with open(join(project_root, 'requirements.txt')) as read_file:
    REQUIRED = read_file.read().splitlines()

with open(join(project_root, 'requirements_test.txt')) as read_file:
    REQUIRED_TEST = read_file.read().splitlines()

with codecs.open(join(project_root, 'README.md'), 'r', 'utf-8') as f:
    long_description = ''.join(f.readlines())

setup(
    name='pya',
    version=version['__version__'],
    description='Python audio coding classes - for dsp and sonification',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=['pya'],
    install_requires=REQUIRED,
    tests_require=REQUIRED_TEST,
    author='Thomas Hermann',
    author_email='thermann@techfak.uni-bielefeld.de',
    keywords=['sonification, sound synthesis'],
    url='https://github.com/thomas-hermann/pya',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis"
    ],
)
