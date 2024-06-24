from ctypes import cdll
cdll.LoadLibrary('/mnt/share/homes/victorvt/envs/cgf_temperature/lib/libstdc++.so.6')
import mf
import paths
import pandas as pd
import argparse
import logging
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = """ Modeler. \
            Example usage: python make_model.py """
    )

    parser.add_argument("measure", help="stunting wasting etc", type=str)
    parser.add_argument("model_identifier", help="model string", type=str)
    parser.add_argument("sex_id", help='1 or 2', type=int)
    parser.add_argument("age_group_id", help='age group', type=int)
    parser.add_argument("model_spec", help='model spec like stunting ~ ihme_loc_id + precip', type=str)
    parser.add_argument("grid_vars", help='list of vars to be binned and added to the grid in order', type=str)
    args = parser.parse_args()

    grid_vars = args.grid_vars.replace('"', '').replace("'", "").split(',')
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    mf.run_model_and_save(args.measure, args.model_identifier, args.sex_id, args.age_group_id, args.model_spec, grid_vars)
    