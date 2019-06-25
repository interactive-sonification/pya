from setuptools import setup
import os
from os.path import join

project_root = os.path.dirname(os.path.abspath(__file__))

with open(join(project_root, 'requirements.txt')) as read_file:
    REQUIRED = read_file.read().splitlines()

with open(join(project_root, 'requirements_test.txt')) as read_file:
    REQUIRED_TEST = read_file.read().splitlines()

setup(
    name='pya',
    version='0.0.1',
    description='python audio coding classes - for dsp and sonification',
    license='MIT',
    packages=['pya'],
    install_requires=REQUIRED,
    tests_require=REQUIRED_TEST,
    author='Thomas Hermann',
    author_email='thermann@techfak.uni-bielefeld.de',
    keywords=['sonification, sound synthesis'],
    url='https://github.com/thomas-hermann/pya'
)
