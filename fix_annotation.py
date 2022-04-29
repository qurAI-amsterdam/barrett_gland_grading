import xml.etree.ElementTree as ET
import re
import os
import argparse
import numpy as np


def get_coords(element):
    """Function to get coordinates of polygon for certain element.
    Parameters:
    element (element): Annotation element with coordinates for polygon inside.
    Returns:
    coords (array): Array with polygon coordinates for element array[[int(x1), int(y1)], ... [int(xN), int(yN)]]
    (Thanks Bart.)
    """
    coords = np.array([[int(re.split(',|\.', coordinates.attrib['X'])[0]), int(re.split(",|\.", coordinates.attrib['Y'])[0])]
                                for coordinates in element.iter('Coordinate')])

    return coords


def fix_annotation_file(annotation_file, verbose=True):
    """Removes all annotations with type polygon, where polygon contains < 3 coordinates.
    A fixed version with (_fixed_) postfix is added to the folder where it was originally found.
    Parameters:
    annotation_file (str): path to the annotation file
    """
    if verbose:
        print('Input xml: {}'.format(annotation_file))

    xml = ET.parse(annotation_file)
    root = xml.getroot()

    # get only the annotations (not the annotation groups)
    annotations = root[0]

    # find the polygons annotations, remove them from the annotations
    for polygon in root.findall(".//Annotation/.[@Type='Polygon']"):
        if len(get_coords(polygon)) < 3:
            if verbose:
                print("Removed a polygon annotation")
            annotations.remove(polygon)

    # write
    output_path = annotation_file.split(".")[0] + "_fixed.xml"
    if verbose:
        print("Writing fixed xml to: {}".format(output_path))
    xml.write(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to the folder where the annotations (xml) are located.")
    parser.add_argument("--verbose", help="Whether to print info.", action="store_true", default=True)
    args = parser.parse_args()

    annotation_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if 'xml' in f]

    if args.verbose:
        print('Found {} annotation files.'.format(len(annotation_files)))

    for f in annotation_files:
        fix_annotation_file(f, args.verbose)

    if args.verbose:
        print('Done!')

