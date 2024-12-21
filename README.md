
# EN-TO-RU translator 

python3 -m venv .env


python train.py

python train.py model=tiny_seq2seq

python train.py model=tiny_seq2seq data=subtitles_small


uvicorn web.main:app --reload --host 0.0.0.0 --port 8000
