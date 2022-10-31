## Start-Up linked Project
### AIPARK - No-Reference Face Image Quality Assessment

- Project Period: Sep. 23 ~ Oct. 25, 2022
- Team: “25 DREAM”  
- Member: Eese Moon, Seongwon Park, Yeongki Baek, Ahyeon Ryang

## Table of Contents 
- [Start-Up linked Project](#start-up-linked-project)
  - [AIPARK - No-Reference Face Image Quality Assessment](#aipark---no-reference-face-image-quality-assessment)
- [Table of Contents](#table-of-contents)
  - [Topic](#topic)
    - [Input Environment](#input-environment)
- [Our suggestion : Combined Advantages of FIQA & IQA (CAFI)](#our-suggestion--combined-advantages-of-fiqa--iqa-cafi)
  - [CAFI Workflow](#cafi-workflow)
  - [Base FIQA Model](#base-fiqa-model)
  - [Remove the background for cropping face only](#remove-the-background-for-cropping-face-only)
  - [Base IQA Model](#base-iqa-model)
  - [Sample Result](#sample-result)

### Topic  
- No-Reference Face Image Quality Assessment  
  학습데이터 수집 시 낮은 품질 이미지를 제외하기 위한 얼굴 전용 평가 지표 생성

#### Input Environment
- Image size 384x384 pixels
- Must not be less than 112x112 pixels


## Our suggestion : Combined Advantages of FIQA & IQA (CAFI)
<img width="480" alt="CAFI_concept" src="https://user-images.githubusercontent.com/103119868/198959453-1ed7c1b0-f90e-4598-8aaa-d68f7accd233.png">  

### CAFI Workflow 
![CAFI](https://user-images.githubusercontent.com/103119868/198959589-02cf3b7e-649b-4a35-8daa-faae508a8c6a.gif)


### Base FIQA Model  
- **SER-FIQ** : Unsupervised estimation of face image quality  
  - Score impact depending on face detection  
  - Simple image quality differences in the same image appear to be indistinguishable  


> Role on CAFI:  
  Used as a filter for recognizable level face images

> *Referece repository: https://github.com/pterhoer/FaceImageQuality*

### Remove the background for cropping face only
> *Referece site: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib*


### Base IQA Model
- **DBCNN** : Additionally used to evaluate the quality of masked face images addition  
> It can be changed according to your preference  
> *Referece repository: https://github.com/chaofengc/IQA-PyTorch*


### Sample Result
<img width="800" alt="CAFI_result" src="https://user-images.githubusercontent.com/103119868/198959424-8d367dac-2d69-4746-93b9-584edd663150.png">

> *Source video: https://www.youtube.com/watch?v=96gLbYFBmKU*