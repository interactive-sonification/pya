from setuptools import setup

with open('requirements.txt') as read_file:
    REQUIRED = read_file.read().splitlines()

setup(
    name='pya',
    version='0.0.1',
    description='python audio coding classes - for dsp and sonification',
    license='MIT',
    packages=['pya'],
    install_requires=REQUIRED,
    author='Thomas Hermann',
    author_email='thermann@techfak.uni-bielefeld.de',
    keywords=['sonification, sound synthesis'],
    url='https://github.com/thomas-hermann/pya'
)
