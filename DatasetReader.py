import os
import glob
import logging
import pandas as pd
import cv2


class DatasetReader(object):
    def __init__(self, db_dir):
        self._dbdir = self.folderpresence(db_dir, create=False)
        self._csvfile = self._get_file('*.csv')
        self._df = None

    def folderpresence(self, folder, create=False):
        if not os.path.isdir(folder):
            if create:
                try:
                    os.makedirs(folder)
                    return folder
                except OSError:
                    raise (OSError, 'The folder {} does not appear to be a valid folder.'.format(folder))
            else:
                print(folder)
                raise (OSError, 'The folder {} does not exist.'.format(folder))
        else:
            return folder

    def _get_file(self, file_format):
        filename = os.path.join(self._dbdir, file_format)
        files = glob.glob(filename)
        if len(files) == 1:
            return files[0]
        else:
            logging.fatal('Multiple files of type {} were found. Cannot choose a file.'.format(file_format))
            raise (ValueError)

    def _read_df(self):
        df = pd.read_csv(self._csvfile)
        return df

    def _list_as_str(self, cols):
        line = ' {}'.format(','.join(cols[:-1]))
        line = '{} and {}'.format(line, cols[-1])
        return line

    def get_annotations(self, query=None):
        self._df = self._read_df()
        cols = self._df.columns.tolist()
        images = set(self._df['filename'].tolist())
        logging.info('Number of images = {}'.format(len(images)))
        if query is not None:
            try:
                df = self._df.query(query)
            except:
                logging.info('The columns in the dataframe are {}'.format(self._list_as_str(cols)))
                raise (ValueError,
                       'The query was not correct. Please check your query and make sure that the dataframe you are trying to query contains the columns mentioned in the query')
        else:
            df = self._df

        cols.remove('filename')
        df = df.groupby('filename')[cols].aggregate(lambda x : tuple(x))
        df = df.reset_index()
        logging.info('Number of images = {}'.format(df.shape[0]))
        return df

    def plot_annotations(self,query=None, df=None, plot_dir=None, plot_cols=None, overwrite=False):
        if plot_dir is None:
            plot_dir = os.path.join(self._dbdir, 'annotations')
        try:
            os.makedirs(plot_dir, exist_ok=True)
        except:
            raise(OSError, 'Cannot create the folder {}'.format(plot_dir))
        if (df is None) and (query is None):
            raise(ValueError, 'One of query and df must be specified.')

        if (df is not None) and (query is not None):
            raise(ValueError, 'Only one of query and df must be specified.')

        if plot_cols is None:
            plot_cols = ['xmin', 'ymin', 'xmax', 'ymax', 'category']

        if df is not None:
            images = df['filename'].tolist()
            if len(images) != len(set(images)):
                raise(ValueError, 'It seems that the dataframe is not grouped. Please use get_annotations() to  group the data.')

        if query is not None:
            df = self.get_annotations(query)

        for index, row in df.iterrows():
            filename = row['filename']
            savefile = os.path.join(plot_dir, os.path.basename(filename))
            if not overwrite:
                if os.path.isfile(savefile):
                    logging.info('The file {} already exists. Skipping'.format(savefile))
                    continue
            image = cv2.imread(filename)
            for xmin, ymin, xmax, ymax, category in zip(row[plot_cols[0]],row[plot_cols[1]], row[plot_cols[2]], row[plot_cols[3]], row[plot_cols[4]]):
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255,0),2)
                cv2.putText(image, category, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

            cv2.imwrite(savefile, image)
            logging.info('The file {} was written with.'.format(savefile))
        return None








