from unittest import TestCase, main, mock
from iisr.data_manager import *
from iisr.preprocessing.representation import HandlerResult
import tempfile
from pathlib import Path
from datetime import datetime
from typing import IO, List


class TestDataManager(TestCase):
    def test_init(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            manager = DataManager(Path(temp_dir_name))
            self.assertTrue(hasattr(manager, 'main_folder'))
            self.assertTrue(manager.main_folder.exists())
            self.assertTrue(manager.main_folder.is_dir())

    def test_save_preproc(self):
        result = mock.Mock()
        dtime = datetime(2015, 1, 1)
        mode_name = 'test'
        short_name = 'short_name'
        preproc_dir = DataManager.PREPROCESSING_FOLDER_NAME
        type(result).dates = mock.PropertyMock(return_value=[dtime])
        type(result).mode_name = mock.PropertyMock(return_value=mode_name)
        result.save_txt.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir_name:
            dirpath = Path(temp_dir_name)
            manager = DataManager(dirpath)
            manager.save_preprocessing_result(result)

            test_filename = mode_name + '_' + short_name + '.dat'
            test_path = str(dirpath / dtime.strftime(DATE_FMT) / preproc_dir / test_filename)

            self.assertTrue(result.save_txt.called)
            self.assertEqual(result.save_txt.call_args[0][0].name, test_path)


if __name__ == '__main__':
    main()