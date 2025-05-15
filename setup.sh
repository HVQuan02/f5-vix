# wget -P Mel-Band-Roformer-Vocal-Model/ https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt
pip install --upgrade -r requirements.txt
# pip install -r Mel-Band-Roformer-Vocal-Model/requirements.txt
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .
export PYTHONPATH="$(pip show f5_tts | grep 'Editable project location' | awk -F ': ' '{print $2}')" # for f5 debug
export GRADIO_SERVER_NAME="0.0.0.0" # for gradio debug
