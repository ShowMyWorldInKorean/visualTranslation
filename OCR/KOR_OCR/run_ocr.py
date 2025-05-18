#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import cv2
import numpy as np
from paddleocr import PaddleOCR

def parse_args():
    parser = argparse.ArgumentParser(description='Run OCR on image')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--output', required=True, help='Directory to save results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 이미지 경로 설정
    image_path = args.image
    image_name = os.path.basename(image_path)
    image_base = os.path.splitext(image_name)[0]
    
    # 모델 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models', 'korean_rec', 'inference')
    dict_path = os.path.join(current_dir, '..', '..', '..', 'paddleocr_repo', 'ppocr', 'utils', 'dict', 'korean_dict.txt')
    
    # PaddleOCR 설정 및 실행
    ocr = PaddleOCR(use_angle_cls=True, lang='korean', 
                   use_gpu=True, det_db_box_thresh=0.5, 
                   det_db_thresh=0.3, det_db_unclip_ratio=1.6,
                   rec_model_dir=model_dir,
                   rec_char_dict_path=dict_path)
    
    # OCR 실행
    result = ocr.ocr(image_path, cls=True)
    
    # 결과 이미지 생성
    image = cv2.imread(image_path)
    vis_image = image.copy()
    
    # 결과 저장을 위한 바운딩 박스 데이터
    bbox_data = {}
    bbox_list = []
    
    # 결과 처리
    if result and result[0]:
        for line in result[0]:
            # 바운딩 박스 좌표와 텍스트
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            # 박스 포인트를 정수로 변환
            points = np.array(bbox, dtype=np.int32)
            
            # x, y, width, height 계산
            x_min = min(point[0] for point in bbox)
            y_min = min(point[1] for point in bbox)
            x_max = max(point[0] for point in bbox)
            y_max = max(point[1] for point in bbox)
            width = x_max - x_min
            height = y_max - y_min
            
            # 바운딩 박스 데이터 저장
            bbox_list.append({
                'box': [[float(p[0]), float(p[1])] for p in bbox],
                'text': text,
                'x': float(x_min),
                'y': float(y_min),
                'width': float(width),
                'height': float(height),
                'confidence': float(confidence)
            })
            
            # 시각화 이미지에 바운딩 박스와 텍스트 그리기
            cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)
            cv2.putText(vis_image, text, (int(x_min), int(y_min) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 상대 경로로 저장
    rel_path = os.path.basename(image_path)
    bbox_data[rel_path] = bbox_list
    
    # 결과 이미지 저장
    output_img_path = os.path.join(args.output, f"{image_base}_ocr.jpg")
    cv2.imwrite(output_img_path, vis_image)
    
    # 바운딩 박스 정보 저장
    bbox_json_path = os.path.join(args.output, f"{image_base}_bbox.json")
    with open(bbox_json_path, 'w', encoding='utf-8') as f:
        json.dump(bbox_data, f, ensure_ascii=False, indent=2)
    
    print(f"OCR completed for {image_path}")
    print(f"Results saved to {output_img_path} and {bbox_json_path}")

if __name__ == "__main__":
    main()