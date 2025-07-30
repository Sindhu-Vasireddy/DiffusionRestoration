from pathlib import Path
import json

import yaml
import argparse


def parse_filename():

    parser = argparse.ArgumentParser(description="Model training.")
    parser.add_argument('-c', '--config_fname',
                         type=str, help="path/to/config_fname.yaml", default="config.yaml")
    args = parser.parse_args()

    return args.config_fname


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle)

def read_yaml(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return yaml.safe_load(handle)