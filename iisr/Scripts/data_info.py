"""
Command line program for display of information about processing results.
"""
from iisr.data_manager import DataManager
import argparse

description = """
Display information about results of processing.
"""

if __name__ == '__main__':
    manager = DataManager()

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--print_folders', action='store_true', default=False,
                        help='print output files folders tree')
    parser.add_argument('--print_id', help='print results with given id')
    # parser.add_argument('ID', type=int, help='ID of processing results')

    args = parser.parse_args()

    if args.print_folders:
        manager.print_folders()

    if args.print_id is not None:
        manager.print_id(args.print_id)