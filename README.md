# annotation_utils
Utilities that are used for assisting with making annotated data.

## Installation
### Install From Github
```console
pip install https://github.com/cm107/annotation_utils/archive/master.zip
```

### Install From Pypi
```console
pip install pyclay-annotation_utils
```

Note: The github package is updated more often than the pypi package.

## Update Build
If you change the code and want to re-build the package, do the following.

```console
./build_package.sh
```

The package can then be installed locally.
```console
pip install -e .
```

## Important Note
* Many of the classes and methods in this repository have been refactored, but have not yet been organized.
* The latest COCO utilities can be found under [annotation_utils/coco/refactored](annotation_utils/coco/refactored).
* The latest Labelme utilities can be found under [annotation_utils/labelme/refactored](annotation_utils/labelme/refactored).
* The latest dataset management utilities can be found under [annotation_utils/dataset/refactored](annotation_utils/dataset/refactored).

## Usage
There is not any official documentation about usage yet, but for usage examples you may refer to the [test folder](test).
* [COCO Utility Usage Examples](test/coco)
* [Labelme Utility Usage Examples](test/labelme)
* [Dataset Utility Usage Examples](test/dataset)