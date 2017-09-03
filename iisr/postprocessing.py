"""
Second stage processing of IISR data.
"""
from datetime import datetime

from iisr.data_manager import DataManager, ID

OPERATIONS = ['power_comparison', 'spectrum']


def run_processing(data_id, start_date, stop_date, operation):
    """
    Manages launch of processing.

    Parameters
    ----------
    data_id: ID
        ID of first stage results.
    start_date: datetime
        Start date of processing.
    stop_date: datetime
        Stop date of processing.
    operation: str
        Mode of operation.
    """
    if operation not in OPERATIONS:
        raise ValueError('Invalid operation: {}'.format(operation))

    manager = DataManager()
    pre_results = manager.get_first_stage_results(data_id=data_id)


def calculate_power_comparison():
    pass


def calculate_spectrum():
    pass
