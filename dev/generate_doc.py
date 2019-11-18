import argparse
import sys
import os 
import subprocess
from os.path import exists

TEMP_FOLDER = 'build/git'
BUILD_FOLDER = f'{TEMP_FOLDER}/build/html/pya'
VERSION_FILE = "versions.html"


def generate(args):
    if not exists(TEMP_FOLDER):
        os.system(f'git clone https://github.com/interactive-sonification/pya.git {TEMP_FOLDER}')
        os.makedirs(BUILD_FOLDER)

    out = subprocess.check_output(f"git -C {TEMP_FOLDER} show-ref --tags -d".split(' '))
    tags = [str(line.split(b' ')[1].split(b'/')[2], 'utf-8') for line in out.split(b'\n')[::-1] if len(line)]
    branches = ['master', 'develop']
    generated = []
    # first, check which documentation to generate
    for t in branches + tags:
        target = 'tags/' + t if t not in branches else t
        os.system(f'git -C {TEMP_FOLDER} checkout {target}')
        if exists(f'{TEMP_FOLDER}/docs/'):
            generated.append(t)
        else:
            print(f'{t} has no docs!')
    
    # put most recent tag to front of list (if exists)
    if len(generated) > len(branches):
        generated = [generated[len(branches)]] + branches + generated[len(branches)+1:]

    # generate documentation; all versions have to be known for the dropdown menu
    for t in generated:
        target = 'tags/' + t if t not in branches else t
        os.system(f'git -C {TEMP_FOLDER} checkout {target} -f')
        os.system(f'cp docs/_templates/* {TEMP_FOLDER}/docs/_templates')
        os.system(f'sphinx-build -b html -D version={t} -A versions={",".join(generated)} {TEMP_FOLDER}/docs {BUILD_FOLDER}/{t}')
    
    # create index html to forward to last tagged version
    with open(f'{BUILD_FOLDER}/index.html', 'w') as fp:
        fp.write(f"""<!DOCTYPE html>
        <html><head>
        <meta http-equiv="refresh" content="0; url={generated[0]}/index.html">
        </head></html>
        """)


if __name__ == "__main__":
    generate(sys.argv[1:])
