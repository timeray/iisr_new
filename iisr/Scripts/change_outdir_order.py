import os
from iisr.data_manager import DataManager
from shutil import move
from iisr.utils import DATE_FMT
import datetime as dt


def main():
    manager = DataManager()
    old_figures_folder_name = 'figures'

    for date_name in os.listdir(manager.main_folder):
        try:
            dt.datetime.strptime(date_name, DATE_FMT)
        except ValueError:
            continue

        data_directory = manager.main_folder / date_name
        if not data_directory.is_dir():
            continue

        pre_proc_folder = data_directory / manager.PREPROCESSING_FOLDER_NAME
        if not pre_proc_folder.exists():
            pre_proc_folder.mkdir(parents=False)

        figures_folder = data_directory / manager.FIGURES_FOLDER_NAME
        if not figures_folder.exists():
            figures_folder.mkdir(parents=False)

        for dirname in os.listdir(data_directory):
            old_directory = data_directory / dirname

            pp_folder_name = manager.PREPROCESSING_FOLDER_NAME
            if dirname != pp_folder_name and str(dirname).startswith(pp_folder_name):
                new_dirname = dirname[(len(pp_folder_name) + 1):]
                new_directory = pre_proc_folder / new_dirname
                print(f'Move {old_directory} to {new_directory}')
                move(old_directory, new_directory)

            if dirname == old_figures_folder_name:
                old_figures_directory = data_directory / dirname
                for name in os.listdir(old_figures_directory):
                    old_path = old_figures_directory / name
                    new_path = figures_folder / name
                    print(f'Move {old_path} to {new_path}')
                    move(old_path, new_path)
                old_figures_directory.rmdir()

            if str(dirname).startswith(old_figures_folder_name):
                new_dirname = dirname[(len(old_figures_folder_name) + 1):]
                new_directory = figures_folder / new_dirname
                print(f'Move {old_directory} to {new_directory}')
                move(old_directory, new_directory)

        if not list(pre_proc_folder.glob('*')):  # empty
            pre_proc_folder.rmdir()

        if not list(figures_folder.glob('*')):
            figures_folder.rmdir()


if __name__ == '__main__':
    main()