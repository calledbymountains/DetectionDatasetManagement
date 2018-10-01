from datasets.caltechpedestrian import CaltechPedestrian
from datasets.bdd100k import BDD100K
from datasets.citypersons import CityPersons
from DatasetReader import DatasetReader
import logging


logging.basicConfig(filename='example.log',level=logging.DEBUG)
base_dir_bdd100k = '/data/stars/share/STARSDATASETS/bdd100k'

for subset in ['train', 'val']:
    db = BDD100K(name='bdd100k-{}'.format(subset), base_dir=base_dir_bdd100k, save_dir='./bdd100k-{}'.format(subset), subset=subset)
    db.writedataframe()
    reader = DatasetReader('./bdd100k-{}'.format(subset))
    df = reader.get_annotations()
    reader.plot_annotations(df=df)


base_dir_caltech = '/data/stars/user/uujjwal/datasets/pedestrian/caltech/caltechall-train'
db = CaltechPedestrian(name='caltechall-train', base_dir=base_dir_caltech, save_dir='./caltechall-train')
db.writedataframe()
reader = DatasetReader('./caltechall-train')
df = reader.get_annotations()
reader.plot_annotations(df=df)


base_dir_caltech = '/data/stars/user/uujjwal/datasets/pedestrian/caltech/caltechall-test'
db = CaltechPedestrian(name='caltechall-test', base_dir=base_dir_caltech, save_dir='./caltechall-test')
db.writedataframe()
reader = DatasetReader('./caltechall-test')
df = reader.get_annotations()
reader.plot_annotations(df=df)

base_dir_citypersons = '/data/stars/user/uujjwal/datasets/pedestrian/cityscapes/leftImg8bit'

for subset in ['train', 'val']:
    db = CityPersons(name='citypersons-{}'.format(subset), base_dir=base_dir_citypersons, save_dir='./citypersons-{}'.format(subset), subset=subset)
    db.writedataframe()
    reader = DatasetReader('./citypersons-{}'.format(subset))
    df = reader.get_annotations()
    reader.plot_annotations(df=df)