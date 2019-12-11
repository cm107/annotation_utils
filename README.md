# annotation_utils
Utilities that are used for assisting with making annotated data.

## Standard Installation (recommended)
```console
pip install https://github.com/cm107/annotation_utils/archive/python3.6.zip
```

## Submodule Installation
See below.

## Explanation
This repository is meant to be used as a submodule.
This repository by itself is a collection of useful methods and classes that relate to creating and managing annotated data.

## Dependencies
In order to be able to use this repository, the following dependencies must be met.

### Pip Dependencies
You can install the pip dependencies to this repository with the following command:
```console
pip install json matplotlib pycocotools shapely numpy
```

Note: It is recommended that you install pip dependencies inside of a virtual environment.

### Directory Structure Dependencies
This repository itself is to be treated as a submodule.
Furthermore, it has other submodule dependencies, which must reside in the parent directory that contains this repository.
* Submodule Dependencies:
    * [logger](https://github.com/cm107/logger.git)
    * [common_utils](https://github.com/cm107/common_utils.git)

### Recommended Directory Structure
* root
    * src
        * submodules
            * logger
            * common_utils
            * annotation_utils

### Setting Up Submodules
Execute the following command from your repository's root directory:

```console
git submodule add https://github.com/cm107/logger.git src/submodules/logger
git submodule add https://github.com/cm107/common_utils.git src/submodules/common_utils
git submodule add https://github.com/cm107/annotation_utils.git src/submodules/annotation_utils
```

## How To Setup Your Submodules After Your Root Repository Is Freshly Pulled
It is recommended that you include the following text into your root repository's README

Add the following alias to your git.
```console
git config --global alias.update '!git pull && git submodule update --init --recursive'
```
Then update your repository.
```console
git update
```