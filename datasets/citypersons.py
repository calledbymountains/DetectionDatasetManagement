from datasets.DatasetBase import DatasetBase
import os
import glob


class CityPersons(DatasetBase):
    def __init__(self, name, base_dir, save_dir, subset):
        super(CityPersons, self).__init__(name, save_dir=save_dir)
        self._basedir = self.folderpresence(base_dir, create=False)
        if subset not in ['train', 'val']:
            raise (ValueError, 'Subset muse be either of "train" or "val".')
        self._subset = subset
        self._imagepath = self.folderpresence(os.path.join(self._basedir, self._subset))
        self._annpath = self.folderpresence(os.path.join(self._basedir, 'annotations_{}'.format(self._subset)))
        self._labelmap = [{"id": 1, "name": "pedestrian"}, {"id": 2, "name": "rider"},
                          {"id": 3, "name": "sitting person"},
                          {"id": 4, "name": "other person"}, {"id": 5, "name": "people group"},
                          {"id": 0, "name": "ignore region"}]

    def imagefiles(self):
        files = glob.glob(os.path.join(self._imagepath, '**', '*.png'), recursive=True)
        return files

    def _read_annotation(self, imagefilename):
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

        basefilename = os.path.splitext(os.path.basename(imagefilename))[0]
        annfile = os.path.join(self._annpath, '{}.txt'.format(basefilename))

        for line in open(annfile, 'r'):
            line = line.strip()
            line = line.split(',')
            if len(line) == 1:
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
                print('Ignoring one annotation in {}. Width or height or both are zero'.format(annfile))
                print('xmin : {}, ymin = {}, width = {}, height = {}'.format(xmin_full, ymin_full, w_full, h_full))
                continue
            a_ratio_vis = w_vis / h_vis
            a_ratio_full = w_full / h_full

            xmax_full = xmin_full + w_full - 1
            ymax_full = ymin_full + h_full - 1
            xmax_vis = xmin_vis + w_vis - 1
            ymax_vis = ymin_vis + h_vis - 1

            annotations['object'].append(self._get_catname(line[0]))
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

    def _get_catname(self, catid):
        catid = int(catid)
        for mapping in self._labelmap:
            if mapping['id'] == catid:
                return mapping['name']

        raise (ValueError, 'category ID {} does not correspond to a valid CityPersons label'.format(catid))

    def annotations(self):
        annotations = []
        for imagefilename in self.imagefiles():
            annotations_single_image = self._read_annotation(imagefilename)
            annotations += annotations_single_image

        return annotations
