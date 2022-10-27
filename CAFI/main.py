import dlib
import cv2
import math
import numpy as np
import glob
import pickle
import pandas as pd
import argparse
from IQA.iqa import get_iqa_score
from extract import Extract
from SER_FIQ.serfiq_example import get_sefiq_score
def extract(args):
    #1 FIQA filter
    serfiq_filter = get_sefiq_score(args.img_folder_path,args.serfiq_threshold)
    #2 Detect filter
    test = Extract(serfiq_filter,args.img_folder_path)
    test.landmarks()
    detect_filter = test.degree()
    #3 IQA filter
    get_iqa_score(detect_filter)

def parse_arg():
    parser = argparse.ArgumentParser(prog='extract final image using CAFI Filter: FIQA > Detect > IQA')

    # parser.add_argument("--run_folder_list", nargs="+", metavar="img_folder", dest="img_folder", help="want to list")
    parser.add_argument("--img_folder_path", type=str, metavar="img_folder_path", dest="img_folder_path", help="input image folder path: /Users/krc")
    parser.add_argument("--serfiq_threshold", type=float, default=0.5, metavar="serfiq_threshold", dest="serfiq_threshold", help="what is threshold above SER-FIQ")

    return parser.parse_args()


def main(argv):
    extract(argv)
if __name__ == "__main__":
    args = parse_arg()
    main(args)