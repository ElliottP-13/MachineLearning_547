import os
import json
from sklearn.model_selection import train_test_split
import shutil

def Good(data_loc, folds):
    path = os.path.join(data_loc, "GOOD_SOUNDS")
    datapoints = {}
    with open(path+"/sounds.json", "r") as file:
        datapoints = json.load(file)
    with open(path+"/takes.json", "r") as file:
        temp = json.load(file)
        for key in temp:
            temp[key].update(datapoints[str(temp[key]["sound_id"])])
    classes = list(temp[key]["instrument"] for key in temp)
    val = []
    train, test = train_test_split(list(temp.keys()), test_size=0.1, random_state=13)
    # train, val = train_test_split(train, test_size=0.11, random_state=13)
    path = data_loc+"/GOOD_MEL_IMAGES"
    for c in set(classes):
        os.makedirs(path + "/train/"+c, exist_ok=True)
        os.makedirs(path + "/test/"+c, exist_ok=True)
        os.makedirs(path + "/validate/"+c, exist_ok=True)
    for t in train:
        shutil.copyfile(path+"/"+temp[t]["filename"].replace(".wav", ".jpg"), path+"/train/"+temp[t]["instrument"]+"/"+temp[t]["pack_filename"].replace(".wav",".jpg"))
    for t in test:
        shutil.copyfile(path + "/" + temp[t]["filename"].replace(".wav", ".jpg"),
                        path + "/test/" + temp[t]["instrument"] +"/"+ temp[t]["pack_filename"].replace(".wav", ".jpg"))
    for v in val:
        shutil.copyfile(path + "/" + temp[v].filename,
                        path + "/validate/" + temp[v]["instrument"] +"/"+ temp[v]["pack_filename"].replace(".wav", ".jpg"))


def NSYNTH(data_loc, folds):
    path = os.path.join(data_loc, "NSYNTH")
    train, test, val = {},{},{}
    with open(path+"/Train/examples.json", "r") as file:
        train = json.load(file)
    with open(path+"/Test/examples.json", "r") as file:
        test = json.load(file)
    with open(path+"/Validate/examples.json", "r") as file:
        val = json.load(file)

    path = os.path.join(data_loc, "NSYNTH_MEL_IMAGES")
    classes = list(train[key]["instrument_family_str"] for key in train)
    for c in set(classes):
        os.makedirs(path + "/train/" + c, exist_ok=True)
        os.makedirs(path + "/test/" + c, exist_ok=True)
        os.makedirs(path + "/validate/" + c, exist_ok=True)
    path = os.path.join(data_loc, "NSYNTH_MEL_IMAGES")
    for t in train:
        shutil.copyfile(path + "/Train/audio" + t+".jpg",
                        path + "/train/" + train[t]["instrument_family_str"] +"/"+ t +".jpg")
    for t in test:
        shutil.copyfile(path + "/Tests/audio" + t+".jpg",
                        path + "/test/" + test[t]["instrument_family_str"] +"/"+ t +".jpg")
    for v in val:
        shutil.copyfile(path + "/Validate/audio" + v+".jpg",
                        path + "/validate/" + val[v]["instrument_family_str"] +"/"+ v +".jpg")


if __name__ == '__main__':
    loc = "/mnt/data1/kwebst_data/data"
    # loc = './data'
    folds = [.9,.1,0]
    # main(loc, split)
    Good(loc, folds)
    NSYNTH(loc, folds)