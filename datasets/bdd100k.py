from datasets.DatasetBase import DatasetBase
import os
import glob
import json
from tqdm import tqdm
import logging
from datasets.bdd100k_labels import Label, labels


class BDD100K(DatasetBase):
    def __init__(self, name, base_dir, save_dir, subset):
        super(BDD100K, self).__init__(name, save_dir)
        if subset not in ['train', 'val']:
            raise (ValueError, 'Subset muse be either of "train" or "val".')
        self._subset = subset
        self._basedir = self.folderpresence(base_dir, create=False)
        self._imagepath = self.folderpresence(os.path.join(self._basedir, 'images', '100k', self._subset))
        self._labelpath = self.folderpresence(
            os.path.join(self._basedir, 'labels', '100k', self._subset))

    def _read_annotations(self, jsonfilename):
        if not os.path.isfile(jsonfilename):
            raise (OSError, 'The file {} could not be found.'.format(jsonfilename))

        with open(jsonfilename, 'r') as jsonfile:
            data = json.load(jsonfile)

        annotations = []
        imagefilename = os.path.join(self._imagepath, '{}.jpg'.format(data['name']))
        attributes = data['attributes']
        if len(data['frames']) != 1:
            logging.debug('In the file {}, the length of frames value is {}'.format(jsonfilename, len(data['frames'])))

        objects = data['frames'][0]['objects']
        logging.debug('Number of objects in {} = {}'.format(imagefilename, len(objects)))

        for obj in objects:
            annotation_single_image = dict(filename=imagefilename)
            annotation_single_image = {**annotation_single_image, **attributes, **obj['attributes']}
            annotation_single_image['category'] = obj['category']

            supercategory, id = self._getinfo(obj['category'])
            annotation_single_image['categoryID'] = id
            annotation_single_image['supercategory'] = supercategory
            if 'box2d' in obj.keys():
                annotation_single_image = {**annotation_single_image, **obj['box2d']}
                annotation_single_image['xmin'] = annotation_single_image.pop('x1')
                annotation_single_image['ymin'] = annotation_single_image.pop('y1')
                annotation_single_image['xmax'] = annotation_single_image.pop('x2')
                annotation_single_image['ymax'] = annotation_single_image.pop('y2')
                annotation_single_image['height'] = annotation_single_image['ymax'] - annotation_single_image[
                    'ymin'] + 1
            else:
                annotation_single_image['xmin'] = -1
                annotation_single_image['ymin'] = -1
                annotation_single_image['xmax'] = -1
                annotation_single_image['ymax'] = -1
                annotation_single_image['height'] = -1

            if 'poly2d' in obj.keys():
                annotation_single_image['poly2d'] = obj['poly2d']
            else:
                annotation_single_image['poly2d'] = []

            annotations.append(annotation_single_image)

        return annotations

    def annotations(self):
        jsonfiles = glob.glob(os.path.join(self._labelpath, '*.json'))
        annotations = []
        for jsonfile in tqdm(jsonfiles):
            annotation_single_image = self._read_annotations(jsonfile)
            annotations += annotation_single_image

        return annotations

    def _getinfo(self, category):
        for info in labels:
            if info.name == category:
                supercategory = info.category
                id = info.id
                return supercategory, id
        else:
            return (ValueError, '{} not found amongs valid BDD100K labels'.format(category))
