import pickle 
import numpy as np
from glob import glob
import joblib
from hydra.utils import to_absolute_path as abs_path
import json


fold = 0
train_wsis = joblib.load(abs_path('liu' + f'/cv{fold}_train_wsi.jb'))
valid_wsis = joblib.load(abs_path('liu' + f'/cv{fold}_valid_wsi.jb'))
test_wsis = joblib.load(abs_path('liu' + f'/cv{fold}_test_wsi.jb'))
Propotion = {}
for i in train_wsis:
    dataset = glob('/home/asanomi/MNISTdata/202203_chemotherapy/mnt1_LEV2/' + '0/' + f'{i}_*/*')
    print(len(dataset))
    dataset2 = glob('/home/asanomi/MNISTdata/202203_chemotherapy/mnt1_LEV2/' + '1/' + f'{i}_*/*')
    print(len(dataset2)) 
    dataset3 = glob('/home/asanomi/MNISTdata/202203_chemotherapy/mnt1_LEV2/' + '2/' + f'{i}_*/*')
    print(len(dataset3)) 
    Propotion[f'{i}'] =  {'0': len(dataset), '1': len(dataset2), '2': len(dataset3)}
    print(Propotion)

for i in valid_wsis:
    dataset = glob('/home/asanomi/MNISTdata/202203_chemotherapy/mnt1_LEV2/' + '0/' + f'{i}_*/*')
    print(len(dataset))
    dataset2 = glob('/home/asanomi/MNISTdata/202203_chemotherapy/mnt1_LEV2/' + '1/' + f'{i}_*/*')
    print(len(dataset2)) 
    dataset3 = glob('/home/asanomi/MNISTdata/202203_chemotherapy/mnt1_LEV2/' + '2/' + f'{i}_*/*')
    print(len(dataset3)) 
    Propotion[f'{i}'] =  {'0': len(dataset), '1': len(dataset2), '2': len(dataset3)}
    print(Propotion)

for i in test_wsis:
    dataset = glob('/home/asanomi/MNISTdata/202203_chemotherapy/mnt1_LEV2/' + '0/' + f'{i}_*/*')
    print(len(dataset))
    dataset2 = glob('/home/asanomi/MNISTdata/202203_chemotherapy/mnt1_LEV2/' + '1/' + f'{i}_*/*')
    print(len(dataset2)) 
    dataset3 = glob('/home/asanomi/MNISTdata/202203_chemotherapy/mnt1_LEV2/' + '2/' + f'{i}_*/*')
    print(len(dataset3)) 
    Propotion[f'{i}'] =  {'0': len(dataset), '1': len(dataset2), '2': len(dataset3)}
    print(Propotion)

tf = open("propotion.json", "w")
json.dump(Propotion,tf)
tf.close()

# with open("propotion.pkl", "wb") as tf:
#     pickle.dump(Propotion,tf)
    # for wsi in train_wsis:
    #     files_list =[
    #             p for p in glob.glob(self.imgs_dir + f"*/{wsi}_*/*.png", recursive=True)
    #             if bool(re_pattern.search(p))
    #                 ]
        