"""
This function processes image img_word_crops, groups them into paragraphs and lines, and outputs the paragraph and line information to a JSON file.

The function takes in several command-line arguments to control the clustering parameters:
- `--a1`: Vertical distance threshold for paragraph clustering (default 0.2)
- `--a2`: Horizontal distance threshold for paragraph clustering (default 0.7)
- `--b1`: Vertical distance threshold for line clustering (default 0.4)
- `--file`: Path to the input JSON file containing the image word_crop information

The function reads in the image word_crop information from the input JSON file, performs clustering and grouping based on spatial relationships, and stores the resulting paragraph and line information in the `patch_info` dictionary. The `patch_info` dictionary is then written to a new JSON file named `para_info.json`.
"""

## exclude_key_words.py
import re
import json
import argparse

import numpy as np

perser = argparse.ArgumentParser()
perser.add_argument("--file", type=str, required=True)
args = perser.parse_args()
file = args.file


tlds = (
    "com", "org", "net", "int", "edu", "gov", "mil", "info", "biz", "name", "museum", "coop", "aero", "xxx", "idv"
)

def exclude(s):
    if re.fullmatch(r'\d+(\.\d+)?', s):
        return True
    if s.startswith("www.") or re.fullmatch(r'(https?://)?(?:[-\w.]|(?:%[\da-fA-F]{2}))+(\.(?:' + '|'.join(tlds) + '))', s):
        return True
    return bool(re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s))

data = json.load(open(file,'r'))
exclude_words = {k: v for k, v in data.items() if not exclude(v["txt"])}

# final_data = {k: v for k, v in data.items() if not exclude(v["txt"])}
json.dump(exclude_words, open("tmp/i_s_info.json",'w'), indent=4)



## detect_para.py

# parser = argparse.ArgumentParser()
# parser.add_argument("--a1", type=float, default=0.2)
# parser.add_argument("--a2", type=float, default=0.7)
# parser.add_argument("--b1", type=float, default=0.4)
# args = parser.parse_args()

# alpha1 = args.a1
# alpha2 = args.a2
# beta1 = args.b1

# file = "tmp/i_s_info.json"

# with open(file, "r") as f:
#     data = json.load(f)

# word_crops = list(data.keys())

# for i in word_crops:
#     data[i]["x1"], data[i]["y1"], data[i]["x2"], data[i]["y2"] = data[i]["bbox"]
#     data[i]["xc"] = (data[i]["x1"] + data[i]["x2"]) / 2
#     data[i]["yc"] = (data[i]["y1"] + data[i]["y2"]) / 2
#     data[i]["w"] = data[i]["x2"] - data[i]["x1"]
#     data[i]["h"] = data[i]["y2"] - data[i]["y1"]

## data -> exclude_words

alpha1 = 0.2
alpha2 = 0.
beta1 = 0.4

beta1 = round(beta1, 1)

word_crops = list(exclude_words.keys())

for i in word_crops:
    exclude_words[i]["x1"], exclude_words[i]["y1"], exclude_words[i]["x2"], exclude_words[i]["y2"] = exclude_words[i]["bbox"]
    exclude_words[i]["xc"] = (exclude_words[i]["x1"] + exclude_words[i]["x2"]) / 2
    exclude_words[i]["yc"] = (exclude_words[i]["y1"] + exclude_words[i]["y2"]) / 2
    exclude_words[i]["w"] = exclude_words[i]["x2"] - exclude_words[i]["x1"]
    exclude_words[i]["h"] = exclude_words[i]["y2"] - exclude_words[i]["y1"]


patch_info = {}
while word_crops:
    img_name = word_crops[0].split("_")[0]
    word_crop_collection = [
        word_crop for word_crop in word_crops if word_crop.startswith(img_name)
    ]
    centroids = {}
    lines = []
    img_word_crops = word_crop_collection.copy()
    para = []
    while img_word_crops:
        clusters = []
        para_words_group = [
            img_word_crops[0],
        ]
        added = [
            img_word_crops[0],
        ]
        img_word_crops.remove(img_word_crops[0])
        ## determining the paragraph
        while added:
            word_crop = added.pop()
            for i in range(len(img_word_crops)):
                word_crop_ = img_word_crops[i]
                if (
                    abs(exclude_words[word_crop_]["yc"] - exclude_words[word_crop]["yc"])
                    < exclude_words[word_crop]["h"] * alpha1
                ):
                    if exclude_words[word_crop]["xc"] > exclude_words[word_crop_]["xc"]:
                        if (exclude_words[word_crop]["x1"] - exclude_words[word_crop_]["x2"]) < exclude_words[
                            word_crop
                        ]["h"] * alpha2:
                            para_words_group.append(word_crop_)
                            added.append(word_crop_)
                    else:
                        if (exclude_words[word_crop_]["x1"] - exclude_words[word_crop]["x2"]) < exclude_words[
                            word_crop
                        ]["h"] * alpha2:
                            para_words_group.append(word_crop_)
                            added.append(word_crop_)
                else:
                    if exclude_words[word_crop]["yc"] > exclude_words[word_crop_]["yc"]:
                        if (exclude_words[word_crop]["y1"] - exclude_words[word_crop_]["y2"]) < exclude_words[
                            word_crop
                        ]["h"] * beta1 and (
                            (
                                (exclude_words[word_crop_]["x1"] < exclude_words[word_crop]["x2"])
                                and (exclude_words[word_crop_]["x1"] > exclude_words[word_crop]["x1"])
                            )
                            or (
                                (exclude_words[word_crop_]["x2"] < exclude_words[word_crop]["x2"])
                                and (exclude_words[word_crop_]["x2"] > exclude_words[word_crop]["x1"])
                            )
                            or (
                                (exclude_words[word_crop]["x1"] > exclude_words[word_crop_]["x1"])
                                and (exclude_words[word_crop]["x2"] < exclude_words[word_crop_]["x2"])
                            )
                        ):
                            para_words_group.append(word_crop_)
                            added.append(word_crop_)
                    else:
                        if (exclude_words[word_crop_]["y1"] - exclude_words[word_crop]["y2"]) < exclude_words[
                            word_crop
                        ]["h"] * beta1 and (
                            (
                                (exclude_words[word_crop_]["x1"] < exclude_words[word_crop]["x2"])
                                and (exclude_words[word_crop_]["x1"] > exclude_words[word_crop]["x1"])
                            )
                            or (
                                (exclude_words[word_crop_]["x2"] < exclude_words[word_crop]["x2"])
                                and (exclude_words[word_crop_]["x2"] > exclude_words[word_crop]["x1"])
                            )
                            or (
                                (exclude_words[word_crop]["x1"] > exclude_words[word_crop_]["x1"])
                                and (exclude_words[word_crop]["x2"] < exclude_words[word_crop_]["x2"])
                            )
                        ):
                            para_words_group.append(word_crop_)
                            added.append(word_crop_)
            img_word_crops = [p for p in img_word_crops if p not in para_words_group]
        ## processing for the line
        while para_words_group:
            line_words_group = [
                para_words_group[0],
            ]
            added = [
                para_words_group[0],
            ]
            para_words_group.remove(para_words_group[0])
            ## determining the line
            while added:
                word_crop = added.pop()
                for i in range(len(para_words_group)):
                    word_crop_ = para_words_group[i]
                    if (
                        abs(exclude_words[word_crop_]["yc"] - exclude_words[word_crop]["yc"])
                        < exclude_words[word_crop]["h"] * alpha1
                    ):
                        if exclude_words[word_crop]["xc"] > exclude_words[word_crop_]["xc"]:
                            if (exclude_words[word_crop]["x1"] - exclude_words[word_crop_]["x2"]) < exclude_words[
                                word_crop
                            ]["h"] * alpha2:
                                line_words_group.append(word_crop_)
                                added.append(word_crop_)
                        else:
                            if (exclude_words[word_crop_]["x1"] - exclude_words[word_crop]["x2"]) < exclude_words[
                                word_crop
                            ]["h"] * alpha2:
                                line_words_group.append(word_crop_)
                                added.append(word_crop_)
                para_words_group = [
                    p for p in para_words_group if p not in line_words_group
                ]
            xc = [exclude_words[word_crop]["xc"] for word_crop in line_words_group]
            idxs = np.argsort(xc)
            patch_cluster_ = [line_words_group[i] for i in idxs]
            line_words_group = patch_cluster_
            x1 = [exclude_words[word_crop]["x1"] for word_crop in line_words_group]
            x2 = [exclude_words[word_crop]["x2"] for word_crop in line_words_group]
            y1 = [exclude_words[word_crop]["y1"] for word_crop in line_words_group]
            y2 = [exclude_words[word_crop]["y2"] for word_crop in line_words_group]
            txt_line = [exclude_words[word_crop]["txt"] for word_crop in line_words_group]
            txt = " ".join(txt_line)
            x = [x1[0]]
            y1_ = [y1[0]]
            y2_ = [y2[0]]
            l = [len(txt_l) for txt_l in txt_line]
            for i in range(1, len(x1)):
                x.append((x1[i] + x2[i - 1]) / 2)
                y1_.append((y1[i] + y1[i - 1]) / 2)
                y2_.append((y2[i] + y2[i - 1]) / 2)
            x.append(x2[-1])
            y1_.append(y1[-1])
            y2_.append(y2[-1])
            line_info = {
                "x": x,
                "y1": y1_,
                "y2": y2_,
                "l": l,
                "txt": txt,
                "word_crops": line_words_group,
            }
            clusters.append(line_info)
        y_ = [clusters[i]["y1"][0] for i in range(len(clusters))]
        idxs = np.argsort(y_)
        clusters_ = [clusters[i] for i in idxs]
        txt = [clusters[i]["txt"] for i in idxs]
        l = [len(t) for t in txt]
        txt = " ".join(txt)
        para_info = {"lines": clusters_, "l": l, "txt": txt}
        para.append(para_info)

    for word_crop in word_crop_collection:
        word_crops.remove(word_crop)
    patch_info[img_name] = {"para": para}

with open("tmp/para_info.json", "w") as f:
    json.dump(patch_info, f, indent=4)

print("Paragraph detection done.")
