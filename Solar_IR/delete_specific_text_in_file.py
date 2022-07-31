import os
import shutil

path = r'F:\solar_IR\SAMPLED'

for dp, dn, fn in os.walk(path):
    for f in fn:
        if 'masked' in f:
            file_path = os.path.join(dp, f)
            print(file_path)
            os.remove(file_path)

