# Release Cycle

## Process

* Update your build tools
```
pip3 install -U setuptools twine wheel
```
* Check/adjust version in `pya/versions.py`
* Update Changelog.md
* Run tox
```
tox
```
* Remove old releases
```
rm dist/*
```
* Build releases
```
python3 setup.py sdist bdist_wheel 
```
* Publish to test.pypi
```
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
* [Check](https://test.pypi.org/project/pya/) the test project page 

* Test install uploaded release
```
pip3 install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pya
```
* Upload the official release
```
twine upload dist/*
```
* [Check](https://pypi.org/project/pya/) the public project page 

* Test install uploaded release
```
pip3 install pya
```
* Draft a release on [Github](https://github.com/thomas-hermann/pya/releases)

## Updating documentation

* Install requirements required to generate the sphinx documenation

```
pip3 install -r dev/requirements_doc.txt
```

* Generate local documentation
```
# this will attempt to open a browser after generation
python3 dev/generate_doc
```

* Generate and publish documentation for all tags and the branches master and develop
```
# Paramater remarks:
# --doctree - generate complete documentation. Ignore local files and checkout complete repo into build
# --publish - publish generated docs to gh-pages instantly. This implicates --doctree
# --clean - remove previous checkouts from build
# --no-show - do not open a browser after generation 

python3 dev/generate_doc --doctree --publish --clean --no-show
```

## Further Readings 

* [using test.PyPI](https://packaging.python.org/guides/using-testpypi/)
* [packaging](https://packaging.python.org/tutorials/packaging-projects/)
