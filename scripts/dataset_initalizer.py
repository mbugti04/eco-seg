import glob

directory = "/media/research/data/flow_1024_512/label_6/*"

files = glob.glob(directory)
print(files)