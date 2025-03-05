import os
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageOps
import json
import re
from kor_ocr import save_all_ocr_results_to_txt

# PaddleOCR 초기화
ocr = PaddleOCR(lang='korean', use_angle_cls=True, use_gpu=True)

# 데이터 경로 설정
IMAGE_PATH = r"E:\OCR\image"
GROUND_TRUTH_PATH = r"E:\OCR\correct"
OUTPUT_TXT_PATH = r"C:\Users\osh\visualTranslation\OCR\KOR_OCR\all_ocr_result.txt"

# OCR 정답에서 "xxx" 제거
def clean_ground_truth(text):
    return ' '.join([word for word in text.split() if word != "xxx"]).strip()

# 정답 파일에서 텍스트 로드
def load_ground_truth(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            annotations = data.get('annotations', [])
            texts = [annotation.get('text', '') for annotation in annotations]
            raw_text = ' '.join(texts).strip()
            return clean_ground_truth(raw_text)
    except Exception:
        return None

# 이미지에서 가장 큰 문자 영역 추출 (ROI 선택)
def extract_largest_text_region(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("L")  # 흑백 변환
            img = ImageOps.invert(img)  # 색상 반전 (텍스트 강조)
            img_array = np.array(img)

            # 이진화
            threshold = 128
            binary_img = np.where(img_array > threshold, 255, 0).astype(np.uint8)

            # 윤곽선 검출
            non_zero_coords = np.column_stack(np.where(binary_img > 0))
            if non_zero_coords.shape[0] == 0:
                return img  # 원본 이미지 반환

            # 바운딩 박스 계산
            x_min, y_min = non_zero_coords.min(axis=0)
            x_max, y_max = non_zero_coords.max(axis=0)

            # ROI 추출 (문자 영역만 자르기)
            return img.crop((y_min, x_min, y_max, x_max))
    except Exception:
        return None

# 평가 실행 (전체 결과를 하나의 딕셔너리에 누적)
def evaluate_paddle_ocr(image_path, ground_truth_path):
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    all_results = {}  # 모든 이미지의 결과 저장 딕셔너리

    if not os.path.exists(image_path) or len(os.listdir(image_path)) == 0:
        print("❌ 처리할 이미지가 없습니다. 경로를 확인하세요.")
        return all_results

    for file in os.listdir(image_path):
        if file.lower().endswith(SUPPORTED_EXTENSIONS):
            try:
                image_file_path = os.path.join(image_path, file)
                json_file_path = os.path.join(ground_truth_path, os.path.splitext(file)[0] + '.json')

                if os.path.exists(json_file_path):
                    ground_truth = load_ground_truth(json_file_path)
                    if ground_truth is None:
                        continue
                else:
                    continue

                # 가장 큰 문자 영역 추출
                cropped_image = extract_largest_text_region(image_file_path)
                if cropped_image is None:
                    continue

                # OCR 인식
                result = ocr.ocr(np.array(cropped_image), cls=True)
                if not result or not result[0]:
                    continue

                paddle_text = ' '.join([line[1][0] for line in result[0]]).strip()

                print(f"📌 파일: {file}")
                print(f"🔍 OCR 결과: {paddle_text}")
                print(f"✅ OCR 정답: {ground_truth}\n")

                # 각 검출 항목을 all_results에 추가
                for idx, detection in enumerate(result[0]):
                    try:
                        bbox_points = detection[0]  # 4개의 점 [[x,y], [x,y], [x,y], [x,y]]
                        xs = [point[0] for point in bbox_points]
                        ys = [point[1] for point in bbox_points]
                        # bbox를 평면 리스트로 변환: [min_x, min_y, max_x, max_y]
                        flat_bbox = [min(xs), min(ys), max(xs), max(ys)]
                        text = detection[1][0]
                        # 키 생성: 파일명(확장자 제거)와 인덱스를 조합하여 "사진이름_박스번호" 형식으로 생성
                        key = f"{os.path.splitext(file)[0]}_{idx}"
                        all_results[key] = {"txt": text, "bbox": flat_bbox}
                    except Exception as e:
                        print(f"검출 항목 처리 중 오류 발생 (인덱스 {idx}): {e}")
                        continue

            except Exception as e:
                print(f"오류 발생: {e}")
                continue

    return all_results

if __name__ == "__main__":
    results = evaluate_paddle_ocr(IMAGE_PATH, GROUND_TRUTH_PATH)
    if results:
        save_all_ocr_results_to_txt(results, OUTPUT_TXT_PATH)
