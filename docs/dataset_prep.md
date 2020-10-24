# Praparing Datasets For Training/Validation
## Navigation
* [Return to Main Menu](../README.md)

## Summary
After creating various collections of COCO datasets, it often becomes necessary to organize them before you can use them for training and validation.
In the context of this document, the goal is to group datasets that are alike into scenarios, combine each scenario's datasets, and then split the scenario into a train dataset and validation dataset.
This document proposes to approaches for doing so:
1. Organizing your dataset folders under their corresponding scenario folder.
    * Combination/splitting into scenario train/val datasets is simplified due to assumptions about the source's directory structure.
    * This approach is less flexible, but is easier to maintain and manage.
    * Refer to prepare_datasets_from_dir
2. Defining the absolute path of each dataset's image directory and annotation path, as well as the corresponding scenario name in an excel file.
    * Combination/splitting into scenario train/val datasets is made more flexible due to the use of absolute paths.
    * This approach is more flexible, but it is necessary to maintain/update an excel sheet in addition to your datasets.

## annotation_utils Installation
This walkthrough assumes that you already have annotation_utils installed to your virtual environment.
The functionality discussed in this document is accurate as of version 0.3.6.

```bash
pip install pyclay-annotation_utils==0.3.6
```

Refer to [README.md](../README.md) for other installation options.

## Source Datasets Explanation
Suppose that we have collections of datasets that we need to merge into scenarios and split into train/val datasets.

For our example, I will use a small bird dataset that I created for debugging/testing purposes.
If you would like to follow along with this walkthrough, you can download [bird_dataset_prep_example_start.zip](https://drive.google.com/file/d/1-3bml3EUhT5l14Z9g3TrGjK_DzBV0QBM/view?usp=sharing) to see the directory structure that we start out with, and you can also download [bird_dataset_prep_example_finish.zip](https://drive.google.com/file/d/13ya1xxbxzmdiVz9_ITKOON65ygdcaQhq/view?usp=sharing) to see the directory stucture that we end up with after we finish the walkthrough.

This is the directory structure of the source root directory that we will be working with.

![starting_state_ss](https://i.imgur.com/0WpZYmY.png)

As you can see from the screenshot, there is no particular restriction on the filename of the images.
Approach 1 requires that the annotation filename of each dataset be the same, but Approach 2 does not.

## Preparing Datasets From A Source Directory (Approach 1)
In this approach, we are combining/splitting scenarios into train and validation datasets based on a source directory with a fixed directory structure.
Let's start by looking at the python script that we will be using.
Save the following script to your bird directory.
```bash=
cd bird
nano prepare_datasets_from_dir.py
```

```python=
from annotation_utils.dataset.dataset_prep import prepare_datasets_from_dir

prepare_datasets_from_dir(
    scenario_root_dir='src_root_dir',
    dst_root_dir='dst_root_dir1',
    annotation_filename='output.json',
    skip_existing=True,
    val_target_proportion=0.05,
    min_val_size=1, max_val_size=20,
    orig_config_save='config_approach1/orig_config.yaml',
    reorganized_config_save='config_approach1/reorganized_config.yaml'
)
```

An explanation of the parameters can be found in [dataset_prep.py](../annotation_utils/dataset/dataset_prep.py).
Notice that the location of our orig_config_save and reorganized_config_save are not inside of dst_root_dir. This is intentional, and it is highly recommended that you do so as well, as it is easier to manage your previous configurations while updating your scenario_root_dir like this.
Let's create a new directory for this approach's configurations.
```bash=
mkdir config_approach1
```
Now we can execute the script.
```bash=
python prepare_datasets_from_dir.py
```

![prepare_datasets_from_dir_exec_ss](https://i.imgur.com/W5mSpMO.png)

Your dst_root_dir should have the following structure.

![dst_root_dir1](https://i.imgur.com/LANppZP.png)

Now we have a single directory that contains all combined scenarios, and each scenario is split into a train dataset and val dataset.

If we look at the contents of config_approach1/orig_config.yaml, it should looks like this.
```yaml=
- collection_dir: /home/clayton/workspace/prj/data_keep/data/dataset/test/bird/src_root_dir/scenario0
  dataset_names:
  - part0
  - part1
  - part2
  dataset_specific:
    ann_format: coco
    ann_path: output.json
    img_dir: .
    tag:
    - part0
    - part1
    - part2
  tag: scenario0
- collection_dir: /home/clayton/workspace/prj/data_keep/data/dataset/test/bird/src_root_dir/scenario1
  dataset_names:
  - part0
  - part1
  - part2
  dataset_specific:
    ann_format: coco
    ann_path: output.json
    img_dir: .
    tag:
    - part0
    - part1
    - part2
  tag: scenario1
```
Furthermore, config_approach1/reorganized_config.yaml should look like this.
```yaml=
- collection_dir: /home/clayton/workspace/prj/data_keep/data/dataset/test/bird/dst_root_dir1
  dataset_names:
  - scenario0
  - scenario1
  dataset_specific:
    ann_format: coco
    ann_path: train/output.json
    img_dir: train
    tag:
    - scenario0_train
    - scenario1_train
  tag: train
- collection_dir: /home/clayton/workspace/prj/data_keep/data/dataset/test/bird/dst_root_dir1
  dataset_names:
  - scenario0
  - scenario1
  dataset_specific:
    ann_format: coco
    ann_path: val/output.json
    img_dir: val
    tag:
    - scenario0_val
    - scenario1_val
  tag: val
```
These configuration files represent the configuration of your datasets in an easy-to-understand and condensed manner, and they can be interfaced with your training script easily using the classes under [dataset_config.py](../annotation_utils/dataset/config/dataset_config.py).

## Preparing Datasets From An Excel Sheet (Approach 2)
In our next approach, we will be combining scenarios and splitting them into train and validation datasets based on an excel sheet. This approach has less restrictions because it defines absolute paths in the excel sheet, but it is harder to maintain than Approach 1.

In our excel sheet, we need to define the Scenario Name, Dataset Name, Image Directory, and Annotation Path of each of our datasets as shown below. If you are following along with the walkthrough, save the excel sheet as datasets.xlsx in your bird directory.

![excel_sheet_ss](https://i.imgur.com/SRJAUQ7.png)

Next let's take a look at the python script that we will be using.
```python=
from annotation_utils.dataset.dataset_prep import prepare_datasets_from_excel

prepare_datasets_from_excel(
    xlsx_path='datasets.xlsx',
    dst_root_dir='dst_root_dir2',
    usecols='A:D', skiprows=None, skipfooter=0, skip_existing=True,
    val_target_proportion=0.05, min_val_size=1, max_val_size=20,
    orig_config_save='config_approach2/orig_config.yaml',
    reorganized_config_save='config_approach2/reorganized_config.yaml',
    show_pbar=True
)
```

Just like in Approach 1, let's create a new directory to store our configurations generated using Approach 2.
```bash=
mkdir config_approach2
```
Now we can execute our python script.
```bash=
python prepare_datasets_from_excel.py
```

![prepare_datasets_from_excel_exec_ss](https://i.imgur.com/BKllqBZ.png)

Notice that the resulting dst_root_dir2 ends up with the same structure as dst_root_dir1, which we created in Approach 1.

![dst_root_dir2_ss](https://i.imgur.com/KxreEx8.png)

The contents of config_approach2/orig_config.yaml and config_approach2/reorganized_config.yaml also end up exactly the same as that of Approach 1.