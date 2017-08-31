"""
Command line program for second stage processing.
Receive type of processing, dates or ID of first stage results, and other option.
"""

import argparse

description = """
Control for second stage processing. Contain multiple types of processing and depend 
on results from first stage.
"""

spectrum_description = """
Processing for signal spectrum.
"""

power_comparison_description = """
Comparison between different power profiles.
"""


def spectrum(args):
    pass


def power_comparison(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers()
    parser.add_argument('start_date', help="start date of processing")
    parser.add_argument('stop_date', help="stop date of processing")
    parser.add_argument('ID', help="ID of first stage results")
    parser.add_argument('--draw', action='store_true', default=False,
                        help="draw all corresponding graphs after processing")

    # Spectrum processing
    spectrum_parser = subparsers.add_parser('spectrum', description=spectrum_description)
    spectrum_parser.set_defaults(func=spectrum)

    # Power comparison processing
    power_comparison_parser = subparsers.add_parser(
        'power_comparison',
        description=power_comparison_description
    )
    power_comparison_parser.set_defaults(func=power_comparison)

    # Parse
    args = parser.parse_args()
    args.func(args)
