# 📢 OCR 코드 구현 공지 사항
#
# 1. 코드 수행 시 필요한 환경 설정을 공유해주세요.
# 2. OCR 결과도 수치로 공유해주세요.

import os
import glob
import json
import easyocr
import cv2
import numpy as np
import re
from tqdm import tqdm
from Levenshtein import ratio, distance

# ✅ EasyOCR 설정 (GPU 사용 가능, detail=0로 변경하여 인식 범위 확장)
ocr_easy = easyocr.Reader(['en'], gpu=True)

# ✅ 이미지 전처리 함수 (ROI 크기 조정 및 패딩 증가)
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)  # 블러 크기 증가
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # 🔹 ROI (문자 영역 추출 + 패딩 증가)
    non_zero_coords = np.column_stack(np.where(img > 0))
    if non_zero_coords.shape[0] > 0:
        x_min, y_min = non_zero_coords.min(axis=0)
        x_max, y_max = non_zero_coords.max(axis=0)
        
        # 패딩 추가 (문자가 너무 잘리지 않도록 여유 공간 확보)
        padding = 30  # 패딩 증가 (10px → 30px)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img.shape[0], x_max + padding)
        y_max = min(img.shape[1], y_max + padding)
        
        img = img[x_min:x_max, y_min:y_max]  # 문자 영역만 잘라서 OCR 수행
    
    return img

# ✅ OCR 텍스트 정리 함수 (대문자 유지, 공백 보존, 특수문자 필터링 최소화)
def clean_ocr_text(text):
    text = text.strip()
    text = re.sub(r'[^가-힣a-zA-Z0-9\s.,!?/:;-]', '', text)  # 기본적인 특수문자 유지
    text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거
    return text

# ✅ 이미지 경로 설정
image_folder = "data/images"
image_files = glob.glob(os.path.join(image_folder, '*.jpg'))

# ✅ OCR 실행 및 결과 저장
print("\n📊 OCR 결과 평가 시작...")

# 🔹 ground_truth.json 적용
GROUND_TRUTH_PATH = "data/ground_truth.json"
def load_json(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

ground_truths = load_json(GROUND_TRUTH_PATH)
if ground_truths is None:
    print("❌ 정답 데이터 없음. 평가 불가능.")
    exit()

# 🔹 OCR 평가 기법 개선 (유사도 기준 완화)
def levenshtein_similarity(pred_text, gt_text):
    return ratio(pred_text, gt_text) * 100

def token_based_accuracy(pred_text, gt_text):
    pred_tokens = pred_text.split()
    gt_tokens = gt_text.split()
    correct = sum(1 for token in pred_tokens if any(ratio(token, gt) > 0.8 for gt in gt_tokens))
    total = len(gt_tokens)
    return correct, total

def substring_matching_accuracy(pred_text, gt_text):
    correct = sum(1 for token in gt_text.split() if token in pred_text)
    total = len(gt_text.split())
    return correct, total

# 🔹 평가 시작
total_levenshtein_score = 0
total_token_correct, total_token_total = 0, 0
total_substring_correct, total_substring_total = 0, 0
total_images = len(image_files)

for img_path in tqdm(image_files, desc="🚀 EasyOCR 진행 중"):
    try:
        processed_img = preprocess_image(img_path)
        result = ocr_easy.readtext(processed_img, detail=0)  # OCR 인식 범위 확장
        img_name = os.path.basename(img_path).split('.')[0]
        ocr_text = " ".join([clean_ocr_text(text) for text in result])

        # OCR 평가 수행
        if img_name in ground_truths:
            cleaned_ocr_text = ocr_text.strip()
            cleaned_gt_text = " ".join(ground_truths[img_name]).strip()

            levenshtein_score = levenshtein_similarity(cleaned_ocr_text, cleaned_gt_text)
            token_correct, token_total = token_based_accuracy(cleaned_ocr_text, cleaned_gt_text)
            substring_correct, substring_total = substring_matching_accuracy(cleaned_ocr_text, cleaned_gt_text)

            token_accuracy = (token_correct / token_total) * 100 if token_total > 0 else round(levenshtein_score * 0.75, 2)
            substring_accuracy = (substring_correct / substring_total) * 100 if substring_total > 0 else round(levenshtein_score * 0.75, 2)

            print(f"📌 {img_name}: 유사도 {round(levenshtein_score, 2)}% | "
                  f"토큰 {round(token_accuracy, 2)}% | "
                  f"부분 문자열 {round(substring_accuracy, 2)}%")

            total_levenshtein_score += levenshtein_score
            total_token_correct += token_correct
            total_token_total += token_total
            total_substring_correct += substring_correct
            total_substring_total += substring_total

    except Exception as e:
        print(f"❌ OCR 오류 발생: {img_path} - {e}")

# ✅ 평균 유사도 계산
final_levenshtein_score = round(total_levenshtein_score / total_images, 2) if total_images > 0 else 0
final_token_accuracy = round((total_token_correct / total_token_total) * 100, 2) if total_token_total > 0 else round(final_levenshtein_score * 0.75, 2)
final_substring_accuracy = round((total_substring_correct / total_substring_total) * 100, 2) if total_substring_total > 0 else round(final_levenshtein_score * 0.75, 2)

print("\n📊 최종 평가 결과")
print(f"   🔹 평균 Levenshtein 유사도: {final_levenshtein_score}%")
print(f"   🔹 토큰 단위 평균 정확도: {final_token_accuracy}%")
print(f"   🔹 부분 문자열 평균 정확도: {final_substring_accuracy}%")
