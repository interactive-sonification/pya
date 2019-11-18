import argparse
import sys
import os 
import subprocess
from os.path import exists

TEMP_FOLDER = 'build/git'
BUILD_FOLDER = f'{TEMP_FOLDER}/build/html'
VERSION_FILE = "versions.html"


def generate(args):
    out = subprocess.check_output("git show-ref --tags -d".split(' '))
    tags = [str(line.split(b' ')[1].split(b'/')[2], 'utf-8') for line in out.split(b'\n')[:-1]]
    branches = ['develop', 'master']
    generated = []
    if not exists(TEMP_FOLDER):
        os.system(f'git clone https://github.com/interactive-sonification/pya.git {TEMP_FOLDER}')
        os.makedirs(BUILD_FOLDER)
    # first, check which documentation to generate
    for t in branches + tags[::-1]:
        if t not in branches:
            t = 'tags/' + t
        os.system(f'git -C {TEMP_FOLDER} checkout {t}')
        if exists(f'{TEMP_FOLDER}/docs/'):
            generated.append(t)
        else:
            print(f'{t} has no docs!')
    # generate documentation; all versions have to be known for the dropdown menu
    for t in generated:
        os.system(f'git -C {TEMP_FOLDER} checkout {t}')
        os.system(f'sphinx-build -b html -D version={t} -A versions={",".join(generated)} {TEMP_FOLDER}/docs {BUILD_FOLDER}/{t}')
    # create index html to forward to last tagged version
    default_version = generated[len(branches)] if len(generated) > len(branches) else generated[-1]
    with open(f'{BUILD_FOLDER}/index.html', 'w') as fp:
        fp.write(f"""<!DOCTYPE html>
        <html><head>
        <meta http-equiv="refresh" content="0; url={default_version}/index.html">
        </head></html>
        """)


if __name__ == "__main__":
    generate(sys.argv[1:])
