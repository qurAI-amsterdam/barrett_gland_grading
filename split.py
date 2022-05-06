import os
from wholeslidedata.dataset import WholeSlideDataSet
from wholeslidedata.source.utils import whole_slide_files_from_folder_factory
from wholeslidedata.source.associations import associate_files
from utils import print_dataset_statistics
import argparse
import numpy as np
from wholeslidedata.source.files import WholeSlideAnnotationFile, WholeSlideImageFile

LABELS = ['NDBE-G', 'LGD-G', 'HGD-G']
SEED = 55


def write_data_yaml(train_associations, val_associations, output_path):
    s = '---\ntraining:'

    for files in train_associations.values():
        image_file = files[WholeSlideImageFile][0].path
        annotation_file = files[WholeSlideAnnotationFile][0].path
        s += '\n\t-\n\t\twsi:\n\t\t\tpath: {}\n\t\twsa:\n\t\t\tpath: {}'.format(image_file, annotation_file)

    s += '\n\nvalidation:'
    for files in val_associations.values():
        image_file = files[WholeSlideImageFile][0].path
        annotation_file = files[WholeSlideAnnotationFile][0].path
        s += '\n\t-\n\t\twsi:\n\t\t\tpath: {}\n\t\twsa:\n\t\t\tpath: {}'.format(image_file, annotation_file)

    print('Writing split yaml file to: {}'.format(output_path))
    with open(output_path, 'w') as out_file:
        out_file.write(s.replace('\t', '    '))


def get_associations_from_folders(folder, datasets):
    total_images = []
    total_annotations = []

    for folder in [os.path.join(folder, dataset) for dataset in datasets]:
        # find all image and annotation files
        print('\nFolder: {}'.format(folder))
        image_files = whole_slide_files_from_folder_factory(folder, 'wsi', excludes=('mask',),
                                                            image_backend='openslide')
        annotation_files = whole_slide_files_from_folder_factory(folder, 'wsa', excludes=('tif',),
                                                                 filters=('fixed',),
                                                                 annotation_parser='asap')

        total_images += image_files
        total_annotations += annotation_files

        print('Found {} image files.'.format(len(image_files)))
        print('Found {} annotation files.'.format(len(annotation_files)))

        # associate image and annotation files
        associations = associate_files(image_files, annotation_files)

        # print dataset statistics
        dataset = WholeSlideDataSet(mode='default', associations=associations, labels=LABELS)
        print_dataset_statistics(dataset)

    # print the total dataset statistics
    total_associations = associate_files(total_images, total_annotations)
    print('\nTotal of annotated images: {}'.format(len(total_associations)))
    total_dataset = WholeSlideDataSet(mode='default', associations=total_associations, labels=LABELS)
    print_dataset_statistics(total_dataset)

    return total_associations


def train_val_split(folder, datasets, output_path, train_percent=0.9):

    # get file keys
    associations = get_associations_from_folders(folder, datasets)
    file_keys = np.array([*associations])
    indexes = np.arange(len(file_keys))

    # shuffle the indexes
    np.random.seed(SEED)
    np.random.shuffle(indexes)

    # split indexes
    n = len(indexes)
    n_train_end = int(train_percent * n)
    train_keys = file_keys[indexes[:n_train_end]]
    val_keys = file_keys[indexes[n_train_end:]]

    train_associations = {file_key: files for file_key, files in associations.items() if file_key in train_keys}
    val_associations = {file_key: files for file_key, files in associations.items() if file_key in val_keys}

    # print train and val set statistics
    train_dataset = WholeSlideDataSet(mode='default', associations=train_associations, labels=LABELS)
    val_dataset = WholeSlideDataSet(mode='default', associations=val_associations, labels=LABELS)

    print('\nTraining dataset')
    print_dataset_statistics(train_dataset, show_all_files=True)
    print('\nValidation dataset')
    print_dataset_statistics(val_dataset, show_all_files=True)

    # write to yaml file
    write_data_yaml(train_associations, val_associations, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to the folder where the datasets are located.")
    parser.add_argument("--datasets", help="Names of the datasets to include to the train and validation split.",
                        default='ASL, LANS', type=str)
    parser.add_argument("--train_percent", help="Percentage used for training, the rest is used for validation.",
                        default=0.9, type=float)
    parser.add_argument("--output_path", help="Path to the folder where the data.yml is stored.",
                        default='/home/mbotros/code/barrett_gland_grading/configs/train_split.yml')
    args = parser.parse_args()
    dataset_names = [dataset for dataset in args.datasets.split(', ')]

    train_val_split(folder=args.input_path, datasets=dataset_names, train_percent=args.train_percent,
                    output_path=args.output_path)

