import os
from os import path,mkdir,walk
from random import randint

def data_format_converter(categoryid,dest_data_path,train_lines, val_lines,image_link_id = 2, categoryid_col_id = 3):
    categoryname = ''.join(categoryname.split()).lower()    # remove spaces
    image_links = []

    for line in train_lines:
        row = line.split()
        if categoryid == int(row[categoryid_col_id]):
            image_links.append("{0},{1},{2},{3}",format(row[0],row[image_link_id],0,1))

    for line in val_lines:
        row = line.split()
        if categoryid == int(row[categoryid_col_id]):
            image_links.append("{0},{1},{2},{3}",format(row[0],row[image_link_id],1,0))
    

    with open(dest_data_path,'w+') as f:
        for line in image_links:
            f.write('{0}\n'.format(line))