import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import urllib
import urllib.request
from importlib import metadata
import modulefinder

COMMANDS = {}

PACKAGE_INDEX_URL = "https://gitlab.ub.uni-bielefeld.de/IchbinLuka/pya_index_test"


def command(name: str):
    def decorator(func):
        COMMANDS[name] = func
        return func

    return decorator


@command("create")
def create(args: argparse.Namespace):
    name = args.name

    if not name.isidentifier():
        print("Invalid project name")
        sys.exit(1)
    
    git_files = [] if args.no_git else [".git/", ".gitignore"]
    files = [
        *git_files,
        "pyproject.toml",
        f"src/{name}/__init__.py",
        "tests/__init__.py"
    ]

    path = os.path.abspath(args.directory)
    
    for file in files:
        if os.path.exists(os.path.join(path, file)):
            print(f"File {file} already exists in {path}. Aborting project creation.")
            sys.exit(1)
    
    if os.path.exists(path):
        # If the directory already exists, ask for confirmation
        files_list = "\n".join(files)
        print(f"This will create the following files in {path}:\n\n{files_list}\n\nDo you want to continue? (y/n)")
        if input().strip().lower() != "y":
            print("Aborting project creation")
            sys.exit(1)

    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs(f"src/{name}", exist_ok=True)
    os.makedirs("tests", exist_ok=True)
    open(f"src/{name}/__init__.py", "w").close()
    open(f"tests/__init__.py", "w").close()
    with open("pyproject.toml", "w") as f:
        # TODO: pya dependency needs to be updated to the main repository before merge
        f.write(
            f"""[build-system]
build-backend = "setuptools.build_meta"
requires = ["setuptools>=61.0"]

[project]
name = "{name}"
version = "0.0.1"
description = "A pya AGen extension"
requires-python = ">=3.10"
readme = "README.md"
authors = []
dependencies = [
    "pya[agen] @ git+https://github.com/IchbinLuka/pya@develop",
]
"""
        )
    if not args.no_git:
        subprocess.run(["git", "init", "--quiet"])
        with urllib.request.urlopen(
            # Download Python .gitignore from GitHub
            "https://raw.githubusercontent.com/github/gitignore/refs/heads/main/Python.gitignore"
        ) as response:
            with open(".gitignore", "w") as f:
                f.write(response.read().decode())
    print(f"Created project {name}")


def resolve_package_url(package_name: str) -> str | None:
    with urllib.request.urlopen(
        # TODO: This should be replaced with a github repo of the  Interactive-Sonification organization
        f"{PACKAGE_INDEX_URL}/-/raw/main/directory.txt"
    ) as response:
        for line in response:
            line = line.decode().strip()
            if line.startswith(package_name):
                return ":".join(line.split(":")[1:]).strip()
    return None


@command("install")
def install(args: argparse.Namespace):
    url = resolve_package_url(args.package_name)
    if url is None:
        print(f"Could not resolve package {args.package_name}")
        sys.exit(1)
    # In uv environments, the pip module is not available, so we use uv pip
    if shutil.which("uv") is not None:
        print("Using uv to install package")
        os.system(f"uv pip install {url}")
    else:
        os.system(f'python -m pip install "{url}"')

@command("get-package")
def get_package(args: argparse.Namespace):
    url = resolve_package_url(args.package_name)
    if url is None:
        print(f"Could not resolve package {args.package_name}")
        sys.exit(1)
    print(url)


@command("publish")
def publish(args: argparse.Namespace):
    popen = subprocess.Popen(["git", "remote"], stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read().decode().splitlines()
    if len(output) == 0:
        print("No remote repository found. Please add a remote repository first.")
        sys.exit(1)
    remote = output[0].strip()
    popen = subprocess.Popen(["git", "remote", "get-url", remote], stdout=subprocess.PIPE)
    popen.wait()
    remote_url = popen.stdout.read().decode().strip()
    if remote_url.startswith("git@"):
        remote_url = remote_url.replace(":", "/")

    head_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    if args.package_name is not None:
        project_name = args.package_name
    else:
        # Else use current directory name as project name
        project_name = pathlib.Path(os.getcwd()).name
    protocol_prefix = "" if remote_url.startswith("https://") else "ssh://"
    package_url = f"{project_name}: git+{protocol_prefix}{remote_url}@{head_hash}"
    print(f"""To publish the current HEAD as '{project_name}', add the following entry to 'directory.txt' in {PACKAGE_INDEX_URL} and open a merge request:

{package_url}""")
    

def main():
    parser = create_parser()
    args = parser.parse_args()
    COMMANDS[args.command](args)

def create_parser():
    parser = argparse.ArgumentParser(description="pya AGen CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    create_parser = subparsers.add_parser("create", help="Create a new pya project")
    create_parser.add_argument("name", help="Name of the project")
    create_parser.add_argument(
        "--no-git", action="store_true", help="Do not initialize a git repository"
    )
    create_parser.add_argument("--directory", type=str, default=".", help="Directory to create the project in")

    get_package_parser = subparsers.add_parser("get-package", help="Looks up a package in the pya package index and returns the URL")
    get_package_parser.add_argument("package-name", help="Name of the package to get the URL for")

    install_parser = subparsers.add_parser("install", help="Helper command to look up a package in the pya package index and install it")
    install_parser.add_argument("package-name", help="Name of the package to install")

    publish_parser = subparsers.add_parser("publish", help="Generates a package URL for the current project based on the git remote URL")
    publish_parser.add_argument("--package-name", help="Name of the package to publish", default=None)

    return parser


if __name__ == "__main__":
    main()
