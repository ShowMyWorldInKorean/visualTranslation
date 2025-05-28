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
    
    # PaddleOCR 설정
    ocr = PaddleOCR(
        lang='korean',
        use_textline_orientation=True
    )

    # OCR 실행
    result = ocr.ocr(image_path, cls=True)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    vis_image = image.copy()

    # 바운딩 박스 저장용 딕셔너리
    bbox_data = {}

    if result and result[0]:
        for idx, line in enumerate(result[0]):
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]

            # 좌표 계산
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            height, width = image.shape[:2]  # 이미지 크기 기준으로 잘못 벗어나지 않게

            # 여유 10픽셀씩 추가
            margin = 10
            x_min = max(min(x_coords) - margin, 0)
            y_min = max(min(y_coords) - margin, 0)
            x_max = min(max(x_coords) + margin, width - 1)
            y_max = min(max(y_coords) + margin, height - 1)
            # 딕셔너리에 저장 (예: "imagebase_0")
            key = f"{image_base}_{idx}"
            bbox_data[key] = {
                "txt": text,
                "bbox": [
                    round(x_min, 2),
                    round(y_min, 2),
                    round(x_max, 2),
                    round(y_max, 2)
                ]
            }

            # 시각화용 바운딩 박스
            points = np.array(bbox, dtype=np.int32)
            cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)
            cv2.putText(vis_image, text, (int(x_min), int(y_min) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
