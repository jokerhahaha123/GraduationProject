import os
import codecs
import shutil
import numpy as np

CURRENT_DIR = os.getcwd()

projects = ["DATA_4"]

NEW_FOLDER = "PROCESSED_DATA_ONLY_JAVA_FILE"
for project in projects:
    new_project_path = os.path.join(CURRENT_DIR, NEW_FOLDER, "Newtonsoft.Json", project)
    cur_project_path = os.path.join(CURRENT_DIR, "collectData\\Newtonsoft.Json", project)
    for r, ds, files in os.walk(cur_project_path):
        for file in files:
            file_path = os.path.join(r, file)

            if file.endswith(".cs"):
                new_r = r.replace(cur_project_path + "\\", "")
                new_path_directory = os.path.join(new_project_path, new_r)
                if not os.path.exists(new_path_directory):
                    os.makedirs(new_path_directory)
                new_file_path = os.path.join(new_path_directory, file)
                print(new_file_path)
                shutil.copy2(file_path, new_file_path)
                # if file.endswith(".java"):
                # 	java_paths.append(file_path)
                # 	num_java +=1
