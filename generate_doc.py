import argparse
import sys
import os 
import subprocess
from os.path import exists

BUILD_FOLDER = 'build/html'
TEMP_FOLDER = 'build/git'
VERSION_FILE = "versions.html"

def generate(args):
    out = subprocess.check_output("git show-ref --tags -d".split(' '))
    tags = ['tags/' + str(line.split(b' ')[1].split(b'/')[2], 'utf-8') for line in out.split(b'\n')[:-1]]
    branches = ['develop', 'master']
    generated = []
    if not exists(TEMP_FOLDER):
        os.system(f'git clone https://github.com/interactive-sonification/pya.git {TEMP_FOLDER}')
    for t in branches + tags:
        os.system(f'git -C {TEMP_FOLDER} checkout {t}')
        if exists(f'{TEMP_FOLDER}/docs/'):
            print(f'sphinx-build -b html -D version={t} {TEMP_FOLDER}/docs {BUILD_FOLDER}/{t}')
            os.system(f'sphinx-build -b html -D version={t} {TEMP_FOLDER}/docs {BUILD_FOLDER}/{t}')
            generated.append(t)
        else:
            print(f'{t} has NOOOOOOO docs')

    version_html_list = ['<li><a href="' + t + '"> ' + t + '</li>' for t in generated]
    with open(f'{BUILD_FOLDER}/{VERSION_FILE}', 'w') as fp:
        fp.write(f"""<!DOCTYPE html>
<html>
<body>
<h2>pya Documentation</h2>
Choose a version:
<ul>
{' '.join(version_html_list)}
</ul>
</body>
</html>
        """)
    
    default_version = generated[-1]
    with open(f'{BUILD_FOLDER}/index.html', 'w') as fp:
        fp.write(f"""<!DOCTYPE html>
        <html><head>
        <meta http-equiv="refresh" content="0; url={default_version}/index.html">
        </head></html>
        """)

if __name__ == "__main__":
    generate(sys.argv[1:])
