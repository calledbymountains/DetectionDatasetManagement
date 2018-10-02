from datasets.caltechpedestrian import CaltechPedestrian
from datasets.bdd100k import BDD100K
from datasets.citypersons import CityPersons
from DatasetReader import DatasetReader
import logging
import os


logging.basicConfig(filename='example.log',level=logging.DEBUG)
base_dir_bdd100k = '/data/stars/share/STARSDATASETS/bdd100k'

for subset in ['train', 'val']:
    db = BDD100K(name='bdd100k-{}'.format(subset), base_dir=base_dir_bdd100k, save_dir='./bdd100k-{}'.format(subset), subset=subset)
    db.writedataframe()
    reader = DatasetReader('./bdd100k-{}'.format(subset))
    df = reader.get_annotations(query='category == "person"')
    reader.plot_annotations(df=df, plot_cols=['xmin', 'ymin', 'xmax', 'ymax', 'category'])


base_dir_caltech = '/data/stars/user/uujjwal/datasets/pedestrian/caltech/caltechall-train'
db = CaltechPedestrian(name='caltechall-train', base_dir=base_dir_caltech, save_dir='./caltechall-train')
db.writedataframe()
reader = DatasetReader('./caltechall-train')
df = reader.get_annotations(query='object == "person"')
reader.plot_annotations(df=df, plot_cols=['xmin_full', 'ymin_full', 'xmax_full', 'ymax_full', 'object'])


base_dir_caltech = '/data/stars/user/uujjwal/datasets/pedestrian/caltech/caltechall-test'
db = CaltechPedestrian(name='caltechall-test', base_dir=base_dir_caltech, save_dir='./caltechall-test')
db.writedataframe()
reader = DatasetReader('./caltechall-test')
df = reader.get_annotations(query='object == "person"')
reader.plot_annotations(df=df, plot_cols=['xmin_full', 'ymin_full', 'xmax_full', 'ymax_full', 'object'])

base_dir_citypersons = '/data/stars/user/uujjwal/datasets/pedestrian/cityscapes/leftImg8bit'

for subset in ['train', 'val']:
    db = CityPersons(name='citypersons-{}'.format(subset), base_dir=base_dir_citypersons, save_dir='./citypersons-{}'.format(subset), subset=subset)
    db.writedataframe()
    reader = DatasetReader('./citypersons-{}'.format(subset))
    df = reader.get_annotations(query='object == "pedestrian" and height_full >= 50 and occlusion <= 0.35')
    reader.plot_annotations(df=df, plot_cols=['xmin_full', 'ymin_full', 'xmax_full', 'ymax_full', 'object'], plot_dir='./citypersons-{}/reasonable'.format(subset))

base_dir_caltech = '/data/stars/user/uujjwal/datasets/pedestrian/caltech'
for subset in ['10x-train', '1x-test']:
    db = CaltechPedestrian(name='caltech{}'.format(subset), base_dir=os.path.join(base_dir_caltech,'caltech{}'.format(subset)), save_dir='./caltech{}'.format(subset))
    db.writedataframe()
    reader = DatasetReader('./caltech{}'.format(subset))
    df = reader.get_annotations(query='height_full >= 50 and occlusion <=0.35 and object == "person"')
    reader.plot_annotations(df=df, plot_cols=['xmin_full','ymin_full','xmax_full','ymax_full','object'], plot_dir='./caltech{}/reasonable'.format(subset))

