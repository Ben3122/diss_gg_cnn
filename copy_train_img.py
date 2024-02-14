import os
import shutil

target_base = "../training_imgs/"

base_path = "../archive"
directories = ["/01/", "/02/", "/03/", "/04/", "/05/", "/06/" , "/07/", "/08/", "/09/" , "/10/"]
text_ending = "cpos.txt"
for dir in directories:
    new_base = base_path + dir
    files = os.listdir(new_base)
    sorted_files = sorted(files)
    for file in sorted_files:
        if file[-3:] == "png":
            shutil.copyfile(new_base + file, target_base + file)
