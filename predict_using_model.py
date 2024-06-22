from ctypes import cdll
cdll.LoadLibrary('/mnt/share/homes/victorvt/envs/cgf_temperature/lib/libstdc++.so.6')
import paths
import argparse
import logging
import pandas as pd
import sys
import fhs_prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = """ Predictor. \
            Example usage: python make_model.py """
    )
    #node_args=["measure", "model_identifier", "lsae_location_id", "scenario", "year", "sex_id", "age_group_id"],


    parser.add_argument("measure", help="stunting wasting etc", type=str)
    parser.add_argument("model_identifier", help="model string", type=str)
    parser.add_argument("fhs_location_id", help="location id in the FHS loc hierarchy", type=int)
    parser.add_argument("scenario", help='ssp126 etc', type=str)
    parser.add_argument("year", help='guess', type=int)
    parser.add_argument("sex_id", help='1 or 2', type=int)
    parser.add_argument("age_group_id", help='age group', type=int)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    fhs_prediction.predict_on_model(args.measure, args.model_identifier, args.fhs_location_id, args.scenario, args.year, args.sex_id, args.age_group_id)
    