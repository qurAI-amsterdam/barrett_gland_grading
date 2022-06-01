import xml.etree.ElementTree as ET
import re
import os
import argparse
import numpy as np
import pandas as pd


def get_coords(element):
    """Function to get coordinates of polygon for certain element.

    Args:
        element (element): Annotation element with coordinates for polygon inside.

    Returns:
        coords (array): Array with polygon coordinates for element array[[int(x1), int(y1)], ... [int(xN), int(yN)]]

    (Thanks Bart)
    """
    coords = np.array(
        [[int(re.split(',|\.', coordinates.attrib['X'])[0]), int(re.split(",|\.", coordinates.attrib['Y'])[0])]
         for coordinates in element.iter('Coordinate')])

    return coords


def fix_annotation_files(files, verbose=True, check=True):
    """Removes all annotations with type polygon, where polygon contains < 3 coordinates.
    A fixed version with (_fixed_) postfix is added to the folder where it was originally found.

    Args:
        files (str): path to the annotation file
        verbose: whether to print info
        check: check only mode (no writing)

    Returns:
        res (pd.DataFrame): Overview in a dataframe of which elements have been removed.
    """
    res = pd.DataFrame(columns=['file', 'annotation', 'coords'])

    for annotation_file in files:
        xml = ET.parse(annotation_file)
        root = xml.getroot()

        # get only the annotations (not the annotation groups)
        annotations = root[0]

        # find the polygons annotations, remove them from the annotations
        for polygon in root.findall(".//Annotation/.[@Type='Polygon']"):
            nr_of_coords = len(get_coords(polygon))
            if nr_of_coords < 3:
                annotations.remove(polygon)
                res = res.concat({'file': annotation_file,
                                  'annotation': polygon.attrib['Name'],
                                  'coords': get_coords(polygon)}, ignore_index=True)
        # write
        if not check:
            output_path = annotation_file.split(".")[0] + "_fixed.xml"
            if verbose:
                print("Writing fixed xml to: {}".format(output_path))
            xml.write(output_path)

    if verbose:
        print(res)

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to the folder where the annotations (xml) are located.")
    parser.add_argument("--verbose", help="Whether to print info.", action="store_true", default=True)
    parser.add_argument("--check", help="Run in check mode, performs no writing.", action="store_true", default=False)
    args = parser.parse_args()

    annotation_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if 'xml' in f]
    annotation_files = [f for f in sorted(annotation_files) if 'fixed' not in f]

    if args.verbose:
        print('Found {} annotation files.'.format(len(annotation_files)))

    results = fix_annotation_files(annotation_files, args.verbose, args.check)

    if args.check:
        csv_out_path = os.path.join(args.input_path, 'polygons_<2_coords.csv')
        print('csv stored in: {}'.format(csv_out_path))
        results.to_csv(csv_out_path)

    if args.verbose:
        print('Done!')
