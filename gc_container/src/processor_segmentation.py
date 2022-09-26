import numpy as np
import torch
import os
from load_models import load_trained_segmentation_model
from rw import READING_LEVEL, WRITING_TILE_SIZE, SegmentationWriter, open_multiresolutionimage_image
from tqdm import tqdm


class SegmentationPipeline:
    def __init__(self, src_path, weights_path):

        # model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model, self.preprocessing = load_trained_segmentation_model(src_path, weights_path)
        self.model.to(self.device)
        self.model.eval()

        # reading/writing config
        self.level = READING_LEVEL
        self.tile_size = WRITING_TILE_SIZE

    def __call__(self, img_path, out_path):
        """ Run the segmentation model on input WSI.
        Todo: Check if it works without tissue segmentation.

        Args:
            img_path: path to the input WSI

        Returns:
            None: writes to an output path.
        """

        # open image
        print('Processing image: {}'.format(img_path))
        image = open_multiresolutionimage_image(path=img_path)

        # get image info
        dimensions = image.getDimensions()
        spacing = image.getSpacing()

        # create writers
        print('Setting up writers')
        segmentation_writer = SegmentationWriter(
            output_path=out_path,
            tile_size=self.tile_size,
            dimensions=dimensions,
            spacing=spacing)

        # loop over image and get tiles
        print("Processing image...")
        for y in tqdm(range(0, dimensions[1], self.tile_size)):
            for x in range(0, dimensions[0], self.tile_size):

                # get an image tile (np.ndarray)
                image_tile = image.getUCharPatch(
                    startX=x, startY=y, width=self.tile_size, height=self.tile_size, level=self.level
                )

                # preprocess the tile
                image_tile = self.preprocessing(image=image_tile)

                # segment the tile
                y_hat = self.model(image_tile).cpu().detach().numpy()
                segmentation_mask = np.argmax(y_hat, axis=1)

                # write the tile segmentation to file
                segmentation_writer.write_segmentation(tile=segmentation_mask, x=x, y=y)

        print('Saving...')
        segmentation_writer.save()
        print('Completed!')


def main():
    """ Pipeline script for running a Docker container on the server or Grand-Challenge.
    """

    # location of input
    img_dir = 'input/images/'
    img_path = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if 'tif' in f][0]
    print('Image path: {}'.format(img_path))

    # location of output
    file_name = img_path.split('/')[-1]
    out_path = os.path.join('/output/images/', file_name)
    print('Output path: {}'.format(out_path))

    # location of src and weights
    src_path = '/opt/algorithm/src/'
    weights_path = '/opt/algorithm/checkpoint.pt'

    # make a pipeline for segmentation
    # pipeline = SegmentationPipeline(src_path=src_path, weights_path=weights_path)
    # pipeline(img_path=img_path, out_path=out_path)


if __name__ == "__main__":
    main()
