from setuptools import setup, find_packages
import annotation_utils

packages = find_packages(
        where='.',
        include=['annotation_utils*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pyclay-annotation_utils',
    version=annotation_utils.__version__,
    description='Utilities that are used for assisting with making annotated data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cm107/annotation_utils",
    author='Clayton Mork',
    author_email='mork.clayton3@gmail.com',
    license='MIT License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'opencv-python>=4.1.1.26',
        'numpy>=1.17.2',
        'Shapely>=1.6.4.post2',
        'matplotlib>=3.1.1',
        'Pillow>=6.1.0',
        'pycocotools>=2.0.0',
        'pylint>=2.4.2',
        'labelme>=3.16.7',
        'PyYAML>=5.1.2',
        'pyqt5>=5.14.1',
        'pyclay-common_utils==0.2.2',
        'pyclay-logger==0.2'
    ],
    python_requires='>=3.7'
)