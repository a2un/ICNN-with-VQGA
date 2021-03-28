from os import walk, path, rename
from shutil import copy
from string import digits

source = './preprocessed_source'
final_path = './all_crop'

new_fnames = {}
for cur_path,_,files in walk('./'):
    for fname in files:
        if path.splitext(fname)[1] == '.txt' and path.splitext(fname)[0].split('_')[1] == 'info':
            for line in open(fname,'r').readlines():
                new_file_name = path.basename(line.split(',')[1])
                obj_name = new_file_name.split('_')[0]
                remove_digits = ''.join([i for i in obj_name if not(i in digits)])

                if not(remove_digits in new_fnames):    
                    new_fnames[remove_digits] = []
                
                if not(new_file_name in new_fnames[remove_digits]):
                    new_fnames[remove_digits].append(new_file_name)

renamed_files = {}
for _,_,files in walk(source):
    for fname in files:
        if  path.splitext(fname)[1] == '.jpg' and \
            len(fname.split('_')) == 3:
            found = False
            index = 0

            if not(fname.split('_')[0] in renamed_files):
                    renamed_files[fname.split('_')[0]] = []

            while not(found) and index < len(new_fnames[fname.split('_')[0]]):
                newfname = new_fnames[fname.split('_')[0]][index]
                
                if not(newfname in renamed_files[fname.split('_')[0]]):
                    copy(path.join(source,fname),final_path)
                    rename(path.join(final_path,fname),path.join(final_path,newfname))
                    renamed_files[fname.split('_')[0]].append(newfname)
                    found = True

                index += 1