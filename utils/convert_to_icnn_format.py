import os
from os import path,mkdir,walk

def data_format_converter(categoryname,source_data_path,dest_data_path):
    categoryname = ''.join(categoryname.split()).lower()    # remove spaces