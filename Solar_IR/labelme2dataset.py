import os

root_path = r'F:\solar_IR\SAMPLED\Label_J'

for dp, dn, fn in os.walk(root_path):
    for f in fn:
        dir_path = os.path.join(dp, f)
        print('Now processing:', dir_path)
        command = 'labelme_json_to_dataset '+dir_path+' -o '+dir_path[:-5]
        os.system(command)
