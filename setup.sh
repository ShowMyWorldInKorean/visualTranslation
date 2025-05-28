#!/bin/bash
# Setup the entire system for visual translation with GPU optimization

## Ensure conda is properly initialized
source $(conda info --base)/etc/profile.d/conda.sh

## ✅ Translation Environment Setup
root_dir=$(pwd)
conda create -n itv2_hf python=3.9 -y
conda activate itv2_hf

# ✅ Ensure CUDA environment is correctly set for GPU
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# ✅ Install essential packages
conda install pip -y
python -m pip install --upgrade pip
python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu118

# ✅ Install other dependencies
python -m pip install nltk sacremoses pandas regex mock "transformers>=4.33.2" mosestokenizer
python -c "import nltk; nltk.download('punkt')"

# ✅ Install prebuilt flash-attn (NO CPU BUILD!)
pip install flash-attn --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

python -m pip install bitsandbytes scipy accelerate datasets --no-cache-dir
python -m pip install sentencepiece

# ✅ Clone and install IndicTransToolkit
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit
python -m pip install --editable ./
pip install scipy
cd $root_dir
conda deactivate

# ## ✅ Scene Text Eraser Environment Setup
# git clone https://github.com/Onkarsus13/Diff_SceneTextEraser.git
# conda create -n scene_text_eraser python=3.9 -y
# conda activate scene_text_eraser

# cd Diff_SceneTextEraser
# pip install -e ".[torch]"
# pip install -e .[all,dev,notebooks]
# pip install jax==0.4.23 jaxlib==0.4.23
# pip install "huggingface_hub<0.26.0"
# cd $root_dir
# conda deactivate

## ✅ SRNet Environment Setup
conda create -n SRNet python=3.9.20 -y
conda activate SRNet
pip install -r SRNet/requirements.txt
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

conda deactivate

## ✅ OCR Environment Setup
echo "Setting up OCR environment..."
conda create -n paddleocr python=3.8 -y
conda activate paddleocr

# Install PaddlePaddle and PaddleOCR
pip install paddlepaddle-gpu==2.6.2 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install "paddleocr>=2.10.0"
pip install opencv-python-headless pillow

# Clone PaddleOCR repository
# if [ ! -d "paddleocr_repo" ]; then
#     echo "Cloning PaddleOCR repository..."
#     git clone https://github.com/PaddlePaddle/PaddleOCR.git paddleocr_repo
# fi

# Setup OCR directory structure
mkdir -p OCR/KOR_OCR/models/korean_rec/inference

# Grant execution permissions
chmod +x OCR/KOR_OCR/run_ocr.py

# Guide for copying model files
echo "OCR environment setup completed."
echo "Please copy the fine-tuned model files to the following path:"
echo "  OCR/KOR_OCR/models/korean_rec/inference/"
echo "Required files: inference.pdiparams, inference.pdiparams.info, inference.pdmodel, inference"

conda deactivate

## ✅ Install Image Processing Libraries
sudo apt update
sudo apt install -y libpango1.0-dev libcairo2-dev imagemagick

## ✅ Final Check: Ensure GPU is recognized
conda activate itv2_hf
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
conda deactivate

# Check GPU in OCR environment
conda activate paddleocr
python -c "import paddle; print('PaddlePaddle GPU Available:', paddle.is_compiled_with_cuda())"
conda deactivate

echo "🎉 Setup complete! Everything is ready to use with GPU acceleration."
echo "To use OCR, add the --ocr flag when running infer.sh"