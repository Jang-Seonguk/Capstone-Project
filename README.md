# Project Title 

**[자동회의록 웹 서비스에 사용될 요약 모델]**  
자동회의록 웹 서비스는 사용자가 회의를 녹음하기만 하면 자동으로 회의를 기록해주고,   
요약해주는 서비스이다. KoBART와 <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat-square&logo=Google Colab&logoColor=white">을 이용하여 요약 모델을 구축하고,   
요약 모델을 이용하여 회의 내용을 요약하고 키워드를 추출한다.


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

This project is licensed under the MIT License - see the [LICENSE](https://gist.github.com/Jang-Seonguk/LICENSE) file for details / 이 프로젝트는 MIT 라이센스로 라이센스가 부여되어 있습니다. 자세한 내용은 LICENSE.md 파일을 참고하세요.



