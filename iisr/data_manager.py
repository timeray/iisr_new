"""
Manages output files and results of processing.
"""
from iisr.representation import FirstStageResults, SecondStageResults
import os
import sys
import configparser


DEFAULT_CONFIG = os.path.join('..', 'general_config.ini')


class DataManager:
    """
    Data manager controls output files of processing.
    """
    FIRST_STAGE_FOLDER_NAME = 'first_stage'
    SECOND_STAGE_FOLDER_NAME = 'second_stage'
    ID_REGISTRY = '.id'

    def __init__(self, path_to_config=DEFAULT_CONFIG):
        """
        Parameters
        ----------
        path_to_config: str
            Path to program basic configuration file.
        """
        config = configparser.ConfigParser()
        config.read(path_to_config)
        self.main_folder = os.path.abspath(config['Common']['path_to_output_storage'])

        if not os.path.exists(self.main_folder):
            os.mkdir(self.main_folder)
        elif not os.path.isdir(self.main_folder):
            raise NotADirectoryError('{}'.format(self.main_folder))

        self.first_stage_folder = os.path.join(self.main_folder,
                                               self.FIRST_STAGE_FOLDER_NAME)
        self.second_stage_folder = os.path.join(self.main_folder,
                                                self.SECOND_STAGE_FOLDER_NAME)
        self.id_registry_path = os.path.join(self.main_folder, self.ID_REGISTRY)

        if not os.path.exists(self.first_stage_folder):
            os.mkdir(self.first_stage_folder)

        if not os.path.exists(self.second_stage_folder):
            os.mkdir(self.second_stage_folder)

        if os.path.exists(self.id_registry_path):
            with open(self.id_registry_path) as id_file:
                self.current_id = int(id_file.readlines()[-1].split(' ')[-1])
        else:
            self.current_id = ID(0)
            self._write_new_id(self.current_id)

    def _write_new_id(self, new_id):
        """
        Write new id to ID_REGISTRY.

        Parameters
        ----------
        new_id: ID
        """
        if not isinstance(new_id, ID):
            raise ValueError('Incorrect ID: {}'.format(new_id))

        with open(self.id_registry_path, 'w') as id_file:
            id_file.write('# Store global identifier. Do not delete or change.\n')
            id_file.write('CURRENT_ID = {}'.format(new_id))

    def get_new_id(self):
        """Return new unique ID"""
        self.current_id += 1
        self._write_new_id(self.current_id)
        return ID(self.current_id)

    def get_results(self, data_id):
        """
        Return first stage results given ID.

        Parameters
        ----------
        data_id: int or ID
            Unique first stage data ID.

        Returns
        -------
        results: Results
        """

    def save_first_stage_results(self, results: FirstStageResults):
        """
        Save first stage processing results.

        Parameters
        ----------
        results: FirstStageResults

        Returns
        -------
        id: ID
            Unique identifier of results.
        """
        new_id = self.get_new_id()
        for result in results.results:
            mode = result.options.mode
            for result_date in result.dates:
                date_path = self.get_path(date=result_date)
                mode_directory = self.mode_directories[mode]
                path = os.path.join(date_path, mode_directory, new_id)
                result.save(path, save_date=result_date)

    def save_second_stage_results(self, results):
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

    def print_id(self, results_id, file=sys.stdout):
        """
        Print results for given id.

        Parameters
        ----------
        results_id: int
        file: stream
        """
        results = self.get_results(results_id)
        print(results, file=file)


class ID(int):
    """Store ID"""


