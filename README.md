# Project Title 

**[자동회의록 웹 서비스에 사용될 요약 모델]**  
자동회의록 웹 서비스는 사용자가 회의를 녹음하기만 하면 자동으로 회의를 기록해주고,   
요약해주는 서비스이다. KoBART와 <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat-square&logo=Google Colab&logoColor=white">을 이용하여 요약 모델을 구축하고,   
요약 모델을 이용하여 회의 내용을 요약하고 키워드를 추출한다.
<br/><br/>



## About

[**BART**](https://arxiv.org/pdf/1910.13461.pdf)(**B**idirectional and **A**uto-**R**egressive **T**ransformers)는 입력 텍스트 일부에 노이즈를 추가하여 이를 다시 원문으로 복구하는 `autoencoder`의 형태로 학습이 됩니다. 한국어 BART(이하 **KoBART**) 는 논문에서 사용된 `Text Infilling` 노이즈 함수를 사용하여 **40GB** 이상의 한국어 텍스트에 대해서 학습한 한국어 `encoder-decoder` 언어 모델입니다. 이를 통해 도출된 `KoBART-base`를 배포합니다.  
(출처 : SKT-AI/KoBART, https://github.com/SKT-AI/KoBART)
<br/><br/>

KoBART Github에 들어가면
* KoBART ChitChatBot
* KoBART Summarization
* NSMC Classification
* KoBART Translation
* LegalQA using SentenceKoBART


위와 같이 나뉘어 있고, 본 프로젝트에서는 KoBART Summarization을 사용하였다.  
Summarization은 seujung님이 모델 패키징을 하셨다.  
<br/><br/>

<p>
  <a href="https://jang-seonguk.github.io/" target="_blank"><img src="https://img.shields.io/badge/Seujung-%23121011?style=flat-square&logo=github&logoColor=white"/></a>
(Seujung님의 Github 주소이다.)
</p>


### Data
- [Dacon 한국어 문서 생성요약 AI 경진대회](https://dacon.io/competitions/official/235673/overview/) 의 학습 데이터를 활용함
- Data 구조
    - Train Data : 34,242
    - Test Data : 8,501

<br/>

### 모델 학습 및 추출

<p>
  <a href="https://colab.research.google.com/drive/1A12_-BNRLzA3rYR_aoL4g5eLz2-Xz-_z?hl=ko" target="_blank"><img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat&logo=Google Colab&logoColor=white"/></a>
</p>

구글 코랩 환경에서 학습 및 추출이 가능합니다.
<br/><br/>


```
!python train.py  --gradient_clip_val 1.0 --max_epochs 1 --default_root_dir /content/drive/MyDrive/Colab_Notebooks/BART --gpus 1 \
--batch_size 4 --num_workers 1 --checkpoint_path /content/drive/MyDrive/Colab_Notebooks/BART \
--train_file /content/KoBART-summarization/data/train.tsv \
--test_file /content/KoBART-summarization/data/test.tsv \
--weights_save_path /content/drive/MyDrive/Colab_Notebooks/BART
```
위 코드를 통해 모델 학습 및 최적화를 진행할 수 있습니다.






<br/><br/>

## Getting Started 



### Prerequisites

```
torch==1.7..1
transformers==4.3.3
pytorch-lightning==1.3.8
streamlit==0.72.0
```


## Running


### <img src="https://img.shields.io/badge/Anaconda-F48220?style=flat-square&logo=Anaconda&logoColor=White"/> Promt 환경에서 실행합니다

```
cd C:\Users\..\Desktop\summary_model
C:\Users\..\Desktop\summary_model> python app.py
```
<br/>

### 실행화면

![image](https://user-images.githubusercontent.com/60394246/154043322-2683ba2c-faea-4bdb-9fdc-388ba5c07aa1.png)


메세지 박스안에 요약하고자 하는 텍스트를 입력하면 됩니다.  
요약 모델은 요약 결과와 키워드를 제공합니다.
<br/><br/><br/><br/>

**원문**

![image](https://user-images.githubusercontent.com/60394246/154043035-d572f7d9-789e-4d6d-a4fc-5cc4ee047679.png)
<br/><br/><br/><br/>

**실행 결과**<br/>

![image](https://user-images.githubusercontent.com/60394246/154044732-307f75fc-b632-4cd4-a886-d5790f8d0ad3.png)


<br/><br/><br/>

## License / 라이센스

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Jang-Seonguk/Capstone-Project/blob/56dc3090c50bd8899ccc59d2ab2cd36506449d51/LICENSE) file for details   
이 프로젝트는 MIT 라이센스로 라이센스가 부여되어 있습니다. 자세한 내용은 LICENSE 파일을 참고하세요.




