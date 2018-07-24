import csv
import glob
import os
import re

from tqdm import tqdm
import pandas as pd
import numpy as no

VARIABLES = ['id', 'full_path', 'subdir', 'gender', 'pattern']
FILE = 'sd04/db_nist.csv'

def read_files_sd04(variables=VARIABLES):
    list_ids = glob.glob('sd04/png_txt/*/*.txt')

    with open(FILE, 'w', newline='') as csv_db:
        writer = csv.writer(csv_db, quoting=csv.QUOTE_ALL)
        writer.writerow(variables)

        for _id in tqdm(list_ids):
            variables = extract_file_path_info(_id)
            with open(_id) as y_txt:
                extract_gender_and_pattern(variables, y_txt)
            writer.writerow(variables)


def extract_gender_and_pattern(variables, y_txt):
    for row, line in enumerate(y_txt.readlines()):
        if row == 0:
            variables.append(re.search('Gender: (.)', line).group(1))
        if row == 1:
            variables.append(re.search('Class: (.)', line).group(1))


def extract_file_path_info(sample):
    variables = []
    variables.append(os.path.splitext(os.path.basename(sample))[0])
    variables.append(os.path.splitext(sample)[0])
    variables.append(os.path.dirname(sample).split(os.sep)[-1])
    return variables


def read_csv_to_dict(length=-1):
    # empty dicts
    mapping = {}
    output = {}
    # location of vars
    pattern_loc = [i for i, var in enumerate(VARIABLES) if var == 'pattern'][0]
    full_path_loc = [i for i, var in enumerate(VARIABLES) if var == 'full_path'][0]
    # mapping of categories set at init
    mapped_integer = 0
    # open file
    with open(FILE) as csv_db:
        # read file
        read_csv = csv.reader(csv_db, delimiter=',')
        for i, row in enumerate(read_csv):
            # skip first row
            if i == 0:
                pass
            # read image information from db
            elif i <= length or length == -1:
                # get category
                cat = row[pattern_loc]
                # set mapping if not present
                if cat not in mapping.keys():
                    mapping[cat] = mapped_integer
                    mapped_integer += 1
                # get mapping
                cat_rec = mapping.get(cat)
                # store to output
                _id = row[full_path_loc]
                _id_san = _id.replace("\\", "/")
                output[_id_san] = cat_rec
            elif i > length:
                break
    return output, mapping


def concat_ids_and_predictions(ids, predictions, look_up, mapping):
    df = pd.DataFrame(predictions, index=ids)
    df.columns = mapping.keys()
    df['pattern'] = None
    inv_map = {v: k for k, v in mapping.items()}
    for _id in ids:
        df.loc[_id, 'pattern'] = inv_map.get(look_up.get(_id))
    df['pred_pattern'] = df[list(mapping.keys())].idxmax(axis=1)
    return df


if __name__ == '__main__':
    read_files_sd04()
