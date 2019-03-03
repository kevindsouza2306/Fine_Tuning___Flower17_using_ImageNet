# Fine_Tuning___Flower17_using_ImageNet<img src="https://img.shields.io/badge/License-MIT-green.svg"  alt="License"/> <a href="https://kevindsouza.ml"><img src="https://img.shields.io/badge/Kevin-D%27souza-blue.svg"  alt="- Kevin Dsouza"/></a>
Trained deep learning model using Transfer learning (fine tuning using keras VGG16 imagenet model) for Flower-17 dataset and gained 95% accuracy

## Download the Pretrained modelusing the below link
<a href="https://drive.google.com/open?id=1pPdPEM6GGbtqcSTT2KC4Xjhb9PxdV6SN">https://drive.google.com/open?id=1pPdPEM6GGbtqcSTT2KC4Xjhb9PxdV6SN</a>

## Download Flower 17 dataset using below link:
<a href="https://drive.google.com/open?id=1zlV6ZEAWA7X4l4pV4VT1nYYuAxA62bn0">https://drive.google.com/open?id=1zlV6ZEAWA7X4l4pV4VT1nYYuAxA62bn0</a>

## Clone and run the flower detect code 

```bash
git clone https://github.com/kevindsouza2306/Fine_Tuning___Flower17_using_ImageNet.git
```
```bash
pip install -r requirements.txt
```
```bash
python detect_flower_17.py -m <location of pretrained model downloaded using the above google drive link> -d <location of the dataset folder>

```
