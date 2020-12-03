# annotation_utils
Utilities that are used for assisting with making annotated data.

## Installation
### Install From Github

```console
# For installing from the master branch (stable)
pip install https://github.com/cm107/annotation_utils/archive/master.zip

# For installing from the development (latest)
pip install https://github.com/cm107/annotation_utils/archive/development.zip
```

Note: If you want to reinstall all dependencies while installing, use the following commands instead:
```console
# For installing from the master branch (stable)
pip install --upgrade --force-reinstall https://github.com/cm107/annotation_utils/archive/master.zip

# For installing from the development (latest)
pip install --upgrade --force-reinstall https://github.com/cm107/annotation_utils/archive/development.zip
```

### Install From Pypi
```console
pip install pyclay-annotation_utils
```

Note: The github package is updated more often than the pypi package.

**Note: In order to overwrite an older version with the latest version, use the following command instead.**
```bash
# Warning: This will automatically upgrade the dependencies too.
pip install --upgrade pyclay-annotation_utils
```

## Install From Source
This package can also be installed locally.

```console
git clone https://github.com/cm107/annotation_utils.git
cd annotation_utils
pip install -e .
```

## Update Build
If you change the code and want to re-build the package, do the following.

```console
./build_package.sh
```

## Important Note
* Outdated/obsolete code has been moved to the [annotation_utils/old](annotation_utils/old) folder as of version 0.2.
* The latest COCO utilities can be found under [annotation_utils/coco](annotation_utils/coco).
* The latest Labelme utilities can be found under [annotation_utils/labelme](annotation_utils/labelme).
* The latest dataset management utilities can be found under [annotation_utils/dataset](annotation_utils/dataset).
* NDDS annotation parsing related classes can be found under [annotation_utils/ndds](annotation_utils/ndds).

## Usage
There is not any official documentation about usage yet, but for usage examples you may refer to the [test folder](test).
* [COCO Utility Usage Examples](test/coco)
* [Labelme Utility Usage Examples](test/labelme)
* [Dataset Utility Usage Examples](test/dataset)
* [NDDS Utility Usage Examples](test/ndds)
* [NDDS to COCO Conversion Examples](test/ndds2coco)

## Documents
The following documents have been written for further explanation.
* [Praparing Datasets For Training/Validation](docs/dataset_prep.md)