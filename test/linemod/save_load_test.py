from annotation_utils.linemod import Linemod_Dataset

dataset = Linemod_Dataset.load_from_path('/home/clayton/workspace/prj/data_keep/data/misc_dataset/new/LINEMOD/cat/train.json')
dataset.save_to_path('temp.json', overwrite=True)
assert Linemod_Dataset.from_dict(dataset.to_dict()) == dataset
print('Test Passed')