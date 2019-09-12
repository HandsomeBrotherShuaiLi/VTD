# Scene Text Detection and Recognition
![](https://img.shields.io/badge/language-python3.6-green.svg)  ![](https://img.shields.io/badge/framework-keras-good.svg)  ![](https://img.shields.io/badge/bulid-passing-red.svg)  ![](https://img.shields.io/badge/author-Shuai_Li-black.svg) ![](https://img.shields.io/badge/Computer_Vision-Detection_and_Recognition-yellow.svg)  
The `VTD` project is on scene text detection and recognition, based on `EAST', 'CTPN' and 'CRNN'.  
## RoadMap
Scene Text Detection is a fast evolving field with new techniques and architectures being published frequently.  
The goal of this project is facilitating the development of such techniques and applications. While constantly improving the quality of code and readme.  
The main functions and models of VTD are as follows:
* Detection: EAST, CTPN and my designed network
* Recognition: CRNN
* GPU_Tracker: track your gpu usage
## Usage
First of all, the `data` package is a preprocess lib to resize or extract the text area.  
'Detection' includes these fancy models, how to use them? start with the `main.py` script of each model. And do not foget to change the data path for your own data.  
`Recognition` only includes the CRNN model, and start with `main.py` to train your network. For details, you can read the arguments of  `DataGenerator.py`.
## Demo
<div align=center><img src="info/figure1.png"/></div>
<div align=center><img src="info/predict_img_26.jpg"/></div>
<div align=center><img src="info/predict_img_27.jpg"/></div>

## Author
* **Shuai Li (李帅)** - *all work*

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
