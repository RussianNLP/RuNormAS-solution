import os


def get_all_files_from_dir(dir_path):
    res = []
    for (dir_path, _, filenames) in os.walk(dir_path):
        for fn in filenames:
            res.append(os.path.join(dir_path, fn))
    return res
