from extract import Extract
import dlib
import cv2
import math
import numpy as np
import glob
import pickle
import pandas as pd
import argparse
from IQA.iqa import iqa_inference

def extract(args):
    test = Extract(args.img_folder_path)
    test.landmarks()
    test.degree()
    iqa_inference(args.img_folder_path)
def parse_arg():
    parser = argparse.ArgumentParser(prog='extract degree from face landmarks')

    # parser.add_argument("--run_folder_list", nargs="+", metavar="img_folder", dest="img_folder", help="want to list")
    parser.add_argument("--img_folder_path", type=str, metavar="img_folder_path", dest="img_folder_path", help="input image folder path: /Users/krc")

    return parser.parse_args()


def main(argv):
    extract(argv)
    # iqa_inference() # IQA infernece , 결과 csv는 score 폴더에 저장됨
if __name__ == "__main__":
    args = parse_arg()
    main(args)