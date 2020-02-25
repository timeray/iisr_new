"""
Manages output files and results of processing.
"""
from collections import namedtuple
from typing import List, Dict

import os
import logging
from pathlib import Path
import configparser
import datetime as dt
from iisr.utils import DATE_FMT
from iisr import IISR_PATH, StdFile


DEFAULT_CONFIG = IISR_PATH / 'general_config.ini'
config = configparser.ConfigParser()
config.read(str(DEFAULT_CONFIG))
CONFIG_MAIN_FOLDER = IISR_PATH / Path(config['Common']['path_to_output_storage'])
if not CONFIG_MAIN_FOLDER.exists():
    CONFIG_MAIN_FOLDER.mkdir()
CONFIG_MAIN_FOLDER = CONFIG_MAIN_FOLDER.resolve()


class DataManager:
    """
    Data manager controls output files of processing.
    """
    PREPROCESSING_FOLDER_NAME = 'pre_proc'
    FIGURES_FOLDER_NAME = 'Figures'
    POSTPROCESSING_FOLDER_NAME = 'post_proc'
    RIOMETER_EXTENSIONS = ('.dat', '.dat.gz')
    RiometerIndex = namedtuple('RiomterIndex', ['date', 'mode'])
    RIOMETER_DATE_FMT = '%Y%m%d'

    def __init__(self, main_folder_path: Path = CONFIG_MAIN_FOLDER,
                 create_main_folder: bool = True,
                 riometer_data_folder: Path = None):
        """

        Args:
            main_folder_path:
        """
        # Main folder is created on demand
        self.main_folder = main_folder_path
        self.created_folders = []  # type: List[Path]
        self.riometer_data_folder = riometer_data_folder
        self.riometer_register = {}  # type: Dict['RiometerIndex', Path]
        if riometer_data_folder is not None:
            if not riometer_data_folder.exists():
                raise ValueError(f'Given riometer data folder ({riometer_data_folder}) not exists')
            self.make_riometer_register()

        if create_main_folder:
            self._check_main_folder()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for folder in self.created_folders:
            while folder.exists() and folder.is_dir() and not list(folder.glob('*')):
                old_folder = folder
                folder = folder.parent
                old_folder.rmdir()

    def _check_main_folder(self):
        if not self.main_folder.exists():
            self.main_folder.mkdir()
        elif not self.main_folder.is_dir():
            raise NotADirectoryError('{}'.format(self.main_folder))

    def make_riometer_register(self):
        logging.info('Create riometer data files registry')
        for dirpath, _, filenames in os.walk(str(self.riometer_data_folder)):
            for filename in filenames:
                if filename.endswith(self.RIOMETER_EXTENSIONS):
                    date_str, mode_str, _ = filename.split('.')[0].split('_')
                    date = dt.datetime.strptime(date_str, self.RIOMETER_DATE_FMT).date()
                    filepath = Path(dirpath) / filename
                    self.riometer_register[self.RiometerIndex(date, mode_str)] = filepath

    def get_preproc_folder_path(self, date: dt.date = None, subfolders: List[str] = None,
                                create_new_folders: bool = True) -> Path:
        if subfolders is None:
            subfolders = []
        return self.get_folder_path(date, subfolders=[self.PREPROCESSING_FOLDER_NAME] + subfolders,
                                    create_new_folders=create_new_folders)

    def get_postproc_folder_path(self, date: dt.date = None, subfolders: List[str] = None,
                                 create_new_folders: bool = True) -> Path:
        if subfolders is None:
            subfolders = []
        return self.get_folder_path(date, subfolders=[self.POSTPROCESSING_FOLDER_NAME] + subfolders,
                                    create_new_folders=create_new_folders)

    def get_riometer_filepath(self, date: dt.date, mode: str) -> Path:
        if self.riometer_data_folder is None:
            raise RuntimeError('Cannot return riometer filepath, because directory was not set')
        assert mode in ['wide', 'narrow']
        mode = 'WD' if mode == 'wide' else 'NR'
        return self.riometer_register[self.RiometerIndex(date, mode)]

    def get_figures_folder_path(self, date: dt.date = None, subfolders: List[str] = None,
                                create_new_folders: bool = True) -> Path:
        if subfolders is None:
            subfolders = []
        return self.get_folder_path(date, subfolders=[self.FIGURES_FOLDER_NAME] + subfolders,
                                    create_new_folders=create_new_folders)

    def get_folder_path(self, date: dt.date = None, subfolders: List[str] = None,
                        create_new_folders: bool = True) -> Path:
        path = self.main_folder
        if date is not None:
            date_str = date.strftime(DATE_FMT)
            path /= date_str

        if subfolders:
            for folder in subfolders:
                path /= folder
        if not path.exists() and create_new_folders:
            path.mkdir(parents=True)
            self.created_folders.append(path)
        return path

    def get_file_path(self, filename: str, date: dt.date = None, subfolders: List[str] = None
                      ) -> Path:
        return self.get_preproc_folder_path(date, subfolders) / filename

    def save_stdfile(self, stdfile: StdFile, filename: str, save_dir_suffix=''):
        save_dirname = 'std'
        if save_dir_suffix:
            save_dirname = save_dirname + '_' + save_dir_suffix

        dirpath = self.main_folder / save_dirname
        if not dirpath.exists():
            dirpath.mkdir(parents=True)

        stdfile.to_file(dirpath / filename)
