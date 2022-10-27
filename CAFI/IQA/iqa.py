import argparse
from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import glob
import os
from pyiqa import create_metric
from tqdm import tqdm
import pandas as pd


def get_iqa_score(detect_filter):
    """Inference demo for pyiqa.
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', type=str, default=None, help='input image/folder path.')
    # parser.add_argument('-r', '--ref', type=str, default=None, help='reference image/folder path if needed.')
    # parser.add_argument(
    #     '--metric_mode',
    #     type=str,
    #     default='FR',
    #     help='metric mode Full Reference or No Reference. options: FR|NR.')
    # parser.add_argument('-m', '--metric_name', type=str, default='PSNR', help='IQA metric name, case sensitive.')
    # parser.add_argument('--save_file', type=str, default=None, help='path to save results.')
    #
    # args = parser.parse_args()

    # choose metric_mode(FR/NR) and name
    # metric_name = args.metric_name.lower()
    metric_mode = "NR"
    metric_name = 'dbcnn'
    # file_path = ''  # insert file or dir path

    # set up IQA model
    iqa_model = create_metric(metric_name, metric_mode=metric_mode)
    metric_mode = iqa_model.metric_mode

    # if os.path.isfile(img_folder_path):
    #     input_paths = [img_folder_path]
    #     # if args.ref is not None:
    #     #     ref_paths = [args.ref]
    # else:
    #     input_paths = sorted(glob.glob(img_folder_path + "/result/" + '*.jpg'))
    #     # if args.ref is not None:
    #     #     ref_paths = sorted(glob.glob(os.path.join(args.ref, '*')))

    # if args.save_file:
    #     sf = open(args.save_file, 'w')

    # load masked image path
    input_paths = detect_filter["masked_path"]
    list_score = []
    file_name = str()
    avg_score = 0
    test_img_num = len(input_paths)
    pbar = tqdm(total=test_img_num, unit='image')
    for idx, img_path in enumerate(input_paths):
        img_name = os.path.basename(img_path)
        if metric_mode == 'FR':
            ref_img_path = ref_paths[idx]
        else:
            ref_img_path = None

        score = iqa_model(img_path, ref_img_path).cpu().item()
        avg_score += score
        pbar.update(1)
        pbar.set_description(f'{metric_name} of {img_name}: {score}')
        pbar.write(f'{metric_name} of {img_name}: {score}')

        list_score.append([img_name, score])
        file_name = img_name

        # if args.save_file:
        #     sf.write(f'{img_name}\t{score}\n')
    pbar.close()

    df_score = pd.DataFrame(data=list_score, columns=['img_name','score'])
    # file_name = file_name.split('_')[1]
    file_name = img_folder_path + "/" + metric_name + '.csv'
    df_score.to_csv(file_name, index=False)

    avg_score /= test_img_num
    if test_img_num > 1:
        print(f'Average {metric_name} score of {img_folder_path} with {test_img_num} images is: {avg_score}')
    # if args.save_file:
    #     sf.close()
    #
    # if args.save_file:
    #     print(f'Done! Results are in {args.save_file}.')
    # else:
    #     print(f'Done!')
