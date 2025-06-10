<h1 align='center'>Show Me the World in My Language: Establishing the First Baseline for Scene-Text to Scene-Text Translation (Korean Version Code)</h1>
<p align='center'>
    <a href="https://icpr2024.org/"><img src="https://img.shields.io/badge/ICPR-2024-4b44ce"></a>
    <a href="https://arxiv.org/abs/2308.03024"><img src="https://img.shields.io/badge/Paper-pdf-red"></a>
    <a href="https://github.com/Bhashini-IITJ/visualTranslation/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue"></a>
    <a href="https://vl2g.github.io/projects/visTrans/"><img src="https://img.shields.io/badge/Project-page-green"></a>
</p>
Implementation of Baseline for Scene Text-to-Scene Text Translation (Eng_Kor version)

<img src="assets/welcome.png" width="100%">

# Release updates:  
- <b>[May 25, 2025]</b> 기존의 레포에서 포크하여 원래는 텍스트 인식, 텍스트 감지는 사람이 라벨링했던 정답 데이터를 활용했으나, OCR 연결해두었습니다(한글 및 영어 연결). 또한 영어-한국어 번역 부분도 옵션으로 설정할 수 있게 수정이 완료 되었습니다.


- <b>[May 25, 2025]</b> Originally, text recognition and text detection were based on human-labeled correct answer data but OCR was implemented by PaddleOCR (connecting Korean and English parts). In addition, the English-Korean translation part has been modified to be set as an option.


# Inference on datasets used 
This release only supports training and inference on datasets used in the paper, i.e., BSTD and ICDAR 2013, and using precomputed scene text detection and recognition. Please follow the below instructions for inference on our VT-Real dataset. For detailed information for specific tasks check the [training](#training) section 

1. Clone the repo and set up the required dependencies
    ```bash
    git clone https://github.com/ShowMyWorldInKorean/visualTranslation.git
    source ./setup.sh
    ```

2. Download the input VT-Real images (which are to be translated) (download details in the [Project page](https://vl2g.github.io/projects/visTrans/)) and put them in folders **source_eng** (ICDAR images) and **source_hin**  (BSTD images) in the project directory.

3. Download the translation checkpoints [eng_hin.model](https://drive.google.com/file/d/1OqloAgsdf-L9hmoeYCW3qrLdtNTQJisx/view?usp=sharing) and [hin_eng.model](https://drive.google.com/file/d/1qb9aUjgGp53lJdfLPUnCVb7mEbd5-gNi/view?usp=sharing) and [eng_kor.model](https://drive.google.com/file/d/1kOY28R3os3xvTvTlsK4FabTnAlTuW6um/view?usp=sharing) and put them in a folder named **model** inside the project directory.

4. We provide precomputed/oracle word-level bounding boxes as json files. (In future release, we plan to integrate scene text detection and recognition implementation to our pipeline). Download these json files from the below table, rename them as engBB.json and hinBB.json for English and Hindi source language datasets, respectively. Then, keep them in the project directory.


| **Source Language** | **Word Bounding Boxes** |
| :---: | :---: |
| Eng | [json file for precomputed word bounding boxes](https://drive.google.com/file/d/1S8ayCLhO2EugF3CLQnHm9J7jJEAq8Hr_/view?usp=drive_link) |
| Hin | [json file for oracle word bounding boxes](https://drive.google.com/file/d/1F_IddWKhw4C4UXOEzH-8a3_4VNqCTias/view?usp=sharing) |


- 여기서 다운로드를 받아도 되나 이미 OCR이 구현되어 있어 해당 정답 데이터를 다운로드 받지 않아도 실행 되는 상태입니다.
- You can still download these data but OCR(text recognition) is implemented so you dont really need to download this data


5. Then, run the following commands to obtain visual translation using our best performing baseline. In both cases a new folder named **output** will be created and the translated images will be saved in it.
  ### Eng &rarr; Hin
  ```bash
  source ./infer.sh -i source_eng -o output -f engBB.json --de
  ```
  ### Hin &rarr; Eng
  Change the checkpoint path in cfg.py file to model/hin_eng.model
  ```bash
  source ./infer.sh -i source_hin  -o output -f hinBB.json --de --hin_eng
  ```
  영_한 번역의 경우 아래의 코드를 수행하세요.
  ### Eng &rarr; Kor
  ```bash
  source ./infer.sh -i source_eng -o output -f engBB.json --kor__eng=false --de
  ```


# Training 
## Dataset generation
The dataset generation script is designed for ImageMagick v6 but can also work with ImageMagick v7, although you may encounter several warnings. The dataset can be generated for either English-to-Hindi (eng-hin) or Hindi-to-English (hin-eng) translations.
### Setup Instructions:
1. Download [this](https://drive.google.com/drive/folders/1Kf4RhqNQ6SP_YJALgWUMG0gvAkbK8S25) folder and add it to your project directory.
2. Unzip all the files within the folder.
3. Install the fonts located in the devanagari.zip file.
   
### Generating the Dataset:
To generate the dataset, run the following command:
```bash
./dataset_gen.sh [ --num_workers <number of loops> --per_worker <number of samples per loop> --hin_eng]
```
Command Options:
--num_workers: Specifies the number of workers for dataset generation. Default: 20.
--per_worker: Specifies the number of samples per loop. Default: 3000.
--hin_eng: Generates a Hindi-to-English (hin-eng) dataset. If not specified, the dataset will be generated for English-to-Hindi (eng-hin).
Note: To generate a dataset for other language pairs, modify the commands in data_gen.py accordingly.

## Training SRNet++

SRNet++ can be trained with the following command:
```bash
conda activate srnet_plus_2
python train_o_t.py
```
change the path of 'data_dir' parameter in cfg.py file if you are using dataset with different path than default.

SRNet++ can be infered with following command lines:
```bash
conda activate srnet_plus_2
python generate_o_t.py
```
please change the path according to your use case. The inputs for the inferece are i_s and i_t. Example given below.
|**i_s**|**i_t**|
|:--:|:--:|
|![](assets/i_s.png)|![](assets/i_t.png)|


# 추론 작동 구조 (inference structure)

## OCR
- paddle OCR을 통해서 한국어, 영어가 인식 가능한 모델을 제작하여 올려두었습니다.
  
- run_ocr.py를 통해서 실행 되며 그 결과를 OCR_merge.py를 통해서 하나로 합쳐서 engBB.json파일이 생성되도록 코드가 짜여있습니다.

## 번역 (Translation)
- 해당 부분에서 번역을 위한 내용을 구비 합니다.
```bash
if [ "$de" = true ]; then
    python exclude_key_words.py --file "$input_file" 
    python detect_para.py
else
    cp "$input_file" tmp/i_s_info.json
    cp "$input_file" tmp/para_info.json
    python form_para_info.py
fi
```

- 해당 코드로 번역을 수행합니다. 이때  git clone https://github.com/VarunGumma/IndicTransToolkit 해당 부분의 코드가 필요합니다.
```bash
if [ "$de" = true ]; then
    if [ "$kor_eng" = true ]; then
        python translate_de.py 
    else
        python translate_de.py --eng_to_kor
    fi
    python form_word_crops.py
else
    if [ "$kor_eng" = true ]; then
        python translate.py 
    else
        python translate.py --eng_to_kor
    fi
fi

```
## 이미지 편집 (SRNet++)
- 해당 코드들을 수행하여 이미지를 편집하고 원본 이미지에 필요한 부분을 삽입하고 출력합니다.
```bash 
python generate_crops.py --folder "$input_folder"

## modifying i_s
python modify_crops.py

## creating i_t
python generate_i_t.py
conda deactivate

# scene text eraser
conda deactivate
conda activate scene_text_eraser
python make_masks.py --folder "$input_folder"
python scene_text_eraser.py --folder "$input_folder"

## generating modified images
python make_output_base.py --folder "$input_folder"

## generating bg
python make_bg.py

## infer srnet_plus_2
conda deactivate
conda activate srnet_plus_2
## 이걸 수행 하기 위해서는 해당 파일과 datagen.py 파일이 필요합니다.
python generate_o_t.py

## blend the crops
python blend_o_t_bg.py

## generate the final output
conda deactivate
conda activate srnet_plus_2
python create_final_images.py --output_folder "$output_folder"
# rm -r tmp


```















---------
## Warning and troubleshooting
- please make sure that imagemagick support png format after the setup.
- Data generation code is written for imagemagickv6. It would work for imagemagickv7 but you will have a lots of warnings. 
## Bibtex (how to cite us)
```
@InProceedings{vistransICPR2024,
    author    = {Vaidya, Shreyas and Sharma, Arvind Kumar and Gatti, Prajwal and Mishra, Anand},
    title     = {Show Me the World in My Language: Establishing the First Baseline for Scene-Text to Scene-Text Translation},
    booktitle = {ICPR},
    year      = {2024},
}
```

## Acknowledgements
1. [SRNet](https://github.com/lksshw/SRNet)
2. [Indic Scene Text Rendering](https://github.com/mineshmathew/IndicSceneTextRendering)
3. [Scene text eraser](https://github.com/Onkarsus13/Diff_SceneTextEraser)
4. [Facebook-m2m](https://huggingface.co/facebook/m2m100_418M)
5. [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)
6. [paddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
## Contact info
In case of any issue/doubt, please raise Github issue and/or write to us: 한영욱 - younguk137@naver.com
