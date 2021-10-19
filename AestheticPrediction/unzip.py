from multiprocessing import Pool
import zipfile
import numpy as np
import os
def f(files):
    with zipfile.ZipFile('/gdrive/MyDrive/AestheticPredictionMLSP/data/images.zip', 'r') as z:
        for f in files:
            name = f.split('/')[-1]
            z.extract(f, path=f'/content')

if __name__ == '__main__':
    with zipfile.ZipFile('/gdrive/MyDrive/AestheticPredictionMLSP/data/images.zip', 'r') as z:
        imgs = [file for file in z.namelist() if ('__MACOSX' not in file) and ('ipynb' not in file) and ('.jpg' in file)]
    imgs = [tuple([img]) for img in np.array_split(imgs,2)]

    with Pool(2) as p:
        p.starmap(f, imgs)
    print(len(os.listdir('/content/images')))