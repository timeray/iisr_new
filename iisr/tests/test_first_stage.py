from unittest import TestCase, main
from iisr.Scripts import first_stage


def setup():
    """Module level setup"""


def teardown():
    """Module level teardown"""


class TestFirstStage(TestCase):
    def test_default_config_run(self):
        # Should run without errors
        first_stage.main('')


if __name__ == '__main__':
    main()
