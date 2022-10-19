from extract import Extract
import dlib
import cv2
import math
import numpy as np
import glob
import pickle
import pandas as pd
import argparse

def extract(args):
    # img_folder = ["2_240p","3_360p"]
    # img_parent_path = "data/"
    test = Extract(args.img_folder, args.img_parent_path)
    test.landmarks()
    test.degree()
def parse_arg():
    parser = argparse.ArgumentParser(prog='extract degree from face landmarks')

    parser.add_argument("--run_folder_list", nargs="+", metavar="img_folder", dest="img_folder", help="want to list")
    parser.add_argument("--img_parent_path", type=str, metavar="img_parent_path", dest="img_parent_path", help="file path")

    return parser.parse_args()


def main(argv):
    extract(argv)

if __name__ == "__main__":
    args = parse_arg()
    main(args)