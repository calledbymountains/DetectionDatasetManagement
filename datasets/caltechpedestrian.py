from datasets.DatasetBase import DatasetBase
import os
import glob
from natsort import natsort_keygen
from tqdm import tqdm


class CaltechPedestrian(DatasetBase):
    def __init__(self, name, base_dir, save_dir):
        super(CaltechPedestrian, self).__init__(name, save_dir=save_dir)
        self._imagedir = self.folderpresence(os.path.join(base_dir, 'images'), create=False)
        self._anndir = self.folderpresence(os.path.join(base_dir, 'annotations'), create=False)
        self._imagelist = self._images()
        self._annlist = self._annotations()

    def _images(self):
        files = glob.glob(os.path.join(self._imagedir, '*.jpg'))
        natsort_key = natsort_keygen(key=lambda x: os.path.splitext(x)[0])
        files.sort(key=natsort_key)
        return files

    def _annotations(self):
        files = glob.glob(os.path.join(self._anndir, '*.txt'))
        natsort_key = natsort_keygen(key=lambda x: os.path.splitext(x)[0])
        files.sort(key=natsort_key)
        return files

    def annotations(self):
        annotations = []
        for imgfile, annfile in tqdm(zip(self._imagelist, self._annlist)):
            annotations_one_image = self._read_annotation(imgfile, annfile)
            annotations += annotations_one_image
        return annotations

    def _read_annotation(self, imagefilename, annfilename):
        if not os.path.isfile(imagefilename):
            raise (OSError, 'The file {} does not exist.'.format(imagefilename))

        if not os.path.isfile(annfilename):
            raise (OSError, 'The file {} does not exist.'.format(annfilename))

        annotations = dict(filename=None,
                           object=[],
                           xmin_vis=[],
                           ymin_vis=[],
                           xmax_vis=[],
                           ymax_vis=[],
                           xmin_full=[],
                           ymin_full=[],
                           xmax_full=[],
                           ymax_full=[],
                           height_full=[],
                           height_vis=[],
                           occlusion=[],
                           aspect_ratio_vis=[],
                           aspect_ratio_full=[])

        for line in open(annfilename, 'r'):
            line = line.strip()
            line = line.split(' ')
            if line[0] == '%':
                continue

            xmin_full, ymin_full, w_full, h_full = list(map(float, line[1:5]))
            xmin_vis, ymin_vis, w_vis, h_vis = list(map(float, line[6:10]))
            if not all([xmin_vis, ymin_vis, w_vis, h_vis]):
                xmin_vis = xmin_full
                ymin_vis = ymin_full
                w_vis = w_full
                h_vis = h_full

            try:
                occlusion = 1. - (w_vis * h_vis) / (w_full * h_full)
            except ZeroDivisionError:
                print('Ignoring one annotation in {}. Width or height or both are zero'.format(annfilename))
                print('xmin : {}, ymin = {}, width = {}, height = {}'.format(xmin_full, ymin_full, w_full, h_full))
                continue
            a_ratio_vis = w_vis / h_vis
            a_ratio_full = w_full / h_full

            xmax_full = xmin_full + w_full - 1
            ymax_full = ymin_full + h_full - 1
            xmax_vis = xmin_vis + w_vis - 1
            ymax_vis = ymin_vis + h_vis - 1

            annotations['object'].append(line[0])
            annotations['xmin_vis'].append(xmin_vis)
            annotations['ymin_vis'].append(ymin_vis)
            annotations['xmax_vis'].append(xmax_vis)
            annotations['ymax_vis'].append(ymax_vis)
            annotations['xmin_full'].append(xmin_full)
            annotations['ymin_full'].append(ymin_full)
            annotations['xmax_full'].append(xmax_full)
            annotations['ymax_full'].append(ymax_full)
            annotations['aspect_ratio_vis'].append(a_ratio_vis)
            annotations['aspect_ratio_full'].append(a_ratio_full)
            annotations['occlusion'].append(occlusion)
            annotations['height_full'].append(h_full)
            annotations['height_vis'].append(h_vis)

        annotations['filename'] = [imagefilename] * len(annotations['xmax_full'])
        annotations_deserialized = []
        for counter in range(len(annotations['filename'])):
            ann = dict()
            for key in annotations.keys():
                ann[key] = annotations[key][counter]
            annotations_deserialized.append(ann)

        return annotations_deserialized
