"""
Manages output files and results of processing.
"""
from typing import IO

import os
import sys
from pathlib import Path
import configparser

from pyasp.stdparse import StdFile

from iisr.utils import DATE_FMT
from iisr import IISR_PATH


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
    POSTPROCESSING_FOLDER_NAME = 'post_proc'

    def __init__(self, main_folder_path: Path = CONFIG_MAIN_FOLDER):
        """

        Args:
            main_folder_path:
        """
        # Main folder is created on demand
        self.main_folder = main_folder_path

    def _check_main_folder(self):
        if not self.main_folder.exists():
            self.main_folder.mkdir()
        elif not self.main_folder.is_dir():
            raise NotADirectoryError('{}'.format(self.main_folder))

    def save_stdfile(self, stdfile: StdFile, filename: str, save_dir_suffix=''):
        self._check_main_folder()

        save_dirname = 'std'
        if save_dir_suffix:
            save_dirname = save_dirname + '_' + save_dir_suffix

        dirpath = self.main_folder / save_dirname
        if not dirpath.exists():
            dirpath.mkdir(parents=True)

        stdfile.to_file(dirpath / filename)

    def save_preprocessing_result(self, result, save_dir_suffix=''):
        """
        Save first stage processing results.

        Parameters
        ----------
        result: FirstStageResults

        Returns
        -------
        id: ID
            Unique identifier of results.
        """
        self._check_main_folder()

        name = result.short_name

        for result_date in result.dates:
            date_str = result_date.strftime(DATE_FMT)
            filename = result.mode_name + '_' + name + '.dat'
            if save_dir_suffix:
                save_dirname = self.PREPROCESSING_FOLDER_NAME + '_' + save_dir_suffix
            else:
                save_dirname = self.PREPROCESSING_FOLDER_NAME

            dirpath = self.main_folder / date_str / save_dirname

            if not dirpath.exists():
                dirpath.mkdir(parents=True)

            with open(str(dirpath / filename), 'w') as file:  # type: IO
                result.save_txt(file, save_date=result_date)

    def save_postprocessing_results(self, results):
        """
        Save second stage processing results.

        Parameters
        ----------
        results: SecondStageResults

        Returns
        -------
        id: ID
            Unique identifier of results.
        """
        self._check_main_folder()

    def print_folders(self, file=sys.stdout):
        """
        Print main folder tree.

        Parameters
        ----------
        file: stream, default sys.stdout
            Stream to write.
        """
        step = '--'
        starting_folder = self.main_folder
        msg = [os.path.basename(starting_folder)]
        nesting = [starting_folder]
        walk = os.walk(starting_folder)
        next(walk)  # skip starting folder
        for dirpath, _, _ in walk:
            basename = os.path.basename(dirpath)
            line = []
            while not dirpath.startswith(nesting[-1]):
                nesting.pop()

            line.append(step * len(nesting))
            line.append(basename)
            msg.append('> '.join(line))
            nesting.append(dirpath)
        print('\n'.join(msg), file=file)
