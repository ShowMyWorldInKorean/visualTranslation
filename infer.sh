#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

m2m=false
hin_eng=false
de=false
eng_kor=false
# kor_eng=false
use_ocr=false  # OCR 플래그 추가

while [ "$1" != "" ]; do
    case $1 in
    "-i")
        shift
        input_folder=$1
        ;;
    "-o")
        shift
        output_folder=$1
        ;;
    "-f")
        shift
        input_file=$1
        ;;
    "--eng_kor")
        eng_kor=true #한영
        ;;
    # "--kor_eng")
    #     kor_eng=true #영한
    #     ;;
    "--de")
        de=true
        ;;
    "--ocr")  # OCR 플래그 추가
        use_ocr=true
        ;;
    esac
    shift
done

mkdir -p tmp

# OCR 처리 추가 (use_ocr 플래그가 true인 경우)
if [ "$use_ocr" = true ]; then
    echo "Running OCR for text detection..."
    
    # OCR conda 환경 활성화
    conda activate paddleocr
    
    # 결과 디렉토리 생성
    mkdir -p tmp/ocr_results
    
    # OCR 정보 출력 (테스트용)
    # if [ -f "./OCR/KOR_OCR/korean_ocr/info.sh" ]; then
    #     bash ./OCR/KOR_OCR/korean_ocr/info.sh
    # fi
    
    # 입력 폴더의 이미지에 OCR 적용
    echo "Scanning images in $input_folder for text..."
    
    # 이미지 파일 찾기
    img_files=$(find "$input_folder" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \))
    
    # OCR 결과를 저장할 JSON 파일 초기화
    echo "{}" > tmp/ocr_bbox.json
    
    # 각 이미지에 대해 OCR 실행
    for img in $img_files; do
        echo "Running OCR on: $img"
        
        # PaddleOCR로 텍스트 인식 실행
        python ./OCR/KOR_OCR/run_ocr.py --image "$img" --output "tmp/ocr_results"
        
        # OCR 결과를 tmp/ocr_bbox.json에 추가
        bbox_file="tmp/ocr_results/$(basename "$img" .jpg)_bbox.json"
        if [ -f "$bbox_file" ]; then
            python -c " "

        echo "Merging OCR results..."
        python ./OCR_merge.py
        echo "✅ OCR merging complete."
        fi
    done
    
    # OCR 결과를 입력 파일로 설정

    
    # OCR 환경 비활성화
    conda deactivate
fi

## paragraph detection
conda activate itv2_hf

if [ "$de" = true ]; then
    python exclude_key_words.py --file "$input_file" 
    python detect_para.py
else
    cp "$input_file" tmp/i_s_info.json
    cp "$input_file" tmp/para_info.json
    python form_para_info.py
fi

## translation
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

## cropping i_s
conda deactivate
conda activate srnet_plus_2
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
python generate_o_t.py

## blend the crops
python blend_o_t_bg.py

## generate the final output
conda deactivate
conda activate srnet_plus_2
python create_final_images.py --output_folder "$output_folder"
# rm -r tmp