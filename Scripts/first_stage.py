"""
Command line program to run first stage processing of IISR data.
"""
import os
import argparse
import configparser

DEFAULT_CONFIG_FILE = os.path.join('..', 'first_stage_config.ini')

config = configparser.ConfigParser()
config.read(DEFAULT_CONFIG_FILE)
print(config.sections())
print(config.get('Common', 'paths_to_process'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser
