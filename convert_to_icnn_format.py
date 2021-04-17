import os
from os import path,mkdir,walk
from random import randint

def data_format_converter():
    image_links = []
    dest_data_path = './icnn/datasets/voc2010_crop/{0}_info.txt'
    train_lines = []
    val_lines = []

    train_lines = open('./data/data_files/train_data.txt','r',encoding='utf-8',errors='ignore').readlines()
    val_lines = open('./data/data_files/val_data.txt','r',encoding='utf-8',errors='ignore').readlines()

    categories = {
        "Person": 1,
        "Chair": 62,
        "Car": 3,
        "Dining_Table": 67,
        "Cup": 47
    }

    for categoryname, categoryid in categories.items():
        for line in train_lines:
            if train_lines.index(line) != 0:
                row = line.split('\t')
                image_links.append("{0},{1},{2},{3}".format(row[0],f'all_crop/{row[0]}',0,int(categoryid in [int(c) for c in row[3].split('---')])))

        for line in val_lines:
            if val_lines.index(line) != 0:
                row = line.split('\t')
                image_links.append("{0},{1},{2},{3}".format(row[0],f'all_crop/{row[0]}',1,int(categoryid in [int(c) for c in row[3].split('---')])))

        with open(dest_data_path.format(categoryname.lower()),'w+') as f:
            for line in image_links:
                f.write('{0}\n'.format(line))

if __name__ == "__main__": data_format_converter()
