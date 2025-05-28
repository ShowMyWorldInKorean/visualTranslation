import os
import json
from glob import glob

ocr_result_dir = "tmp/ocr_results"
merged_output_path = "engBB.json"

# 모든 *_bbox.json 파일 찾기
bbox_files = sorted(glob(os.path.join(ocr_result_dir, "*_bbox.json")))

merged = {}

for bbox_file in bbox_files:
    with open(bbox_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # data가 {"100_0": {...}, "100_1": {...}} 형식일 때
        if all(isinstance(v, dict) and "txt" in v for v in data.values()):
            merged.update(data)
        # data가 {"100": {"100_0": {...}, "100_1": {...}}} 형식일 때
        else:
            for image_id, entries in data.items():
                merged.update(entries)

# 결과 저장
with open(merged_output_path, 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"✅ OCR 결과 병합 완료: {merged_output_path}")
