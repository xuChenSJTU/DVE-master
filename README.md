# DVE-master  
This is a tensorflow implementation of our TWEB 2019 paper:  
Decoupled Variational Embedding for Signed Directed Networks.  
You can access the paper through:  https://arxiv.org/abs/2008.12450

Requirements:  
tensorflow-gpu==1.7.0;  
pandas==0.25.0;  
numpy==1.15.2;  
scipy==1.1.0;  
sklearn==0.20.0;  

How to run:  

First, you need to download the data in:   
Google Drive:  https://drive.google.com/file/d/12d1oZDT338ez9fAVSLgSwQKg7lMiGHCy/view?usp=sharing  
For users who are not accessible to google drive, we have:  
Baidu Disk: https://pan.baidu.com/s/1fpZFThvqZ6SoHt1g2lSkaw  password:ok18   
When you finished the download, just merge the two folders (the downloaded one and this online one, the downloaded one has the complete files. When merge, replace the existing files.)   

If you do not want to process the datasets, just skip Step 1 to Step2. We have provided the processed data under /data   

Step1. Data processing:  
(1) The used three datasets are Epinions, Slashdot and Wiki that are provided in /graph.  
Due to our limited computational resources, we randomly sample a subset of the original data and the used datasets are: epinions_small.txt, slashdot_small.txt and wiki_small.txt in /graph.  
(2) Run preprocess.py to get epinions_small.txt, slashdot_small.txt and wiki_small.txt  
(3) Run split_train_links to get train data for our model  

Step2. Codes of Methods:  
(1) Note the file contains three methods: DVE_main.py, DE_main.py and BPWR_main.py. The last two are two variants of the proposed DVE method. Each one has a xxxx_model.py file that contains the definition of model.  
(2) Run each model. You can just run a model by:  

CUDA_VISIBLE_DEVICES=gpu_num python DVE_main.py  

The default dataset is epinions_small and other hyper-parameters can be changed according to the demonstration in our paper.  

If you find this paper and codes are useful, please cite our paper. Thank you.  

@inproceedings{xu2019decoupled,  
  title={Decoupled Variational Embedding for Signed Directed Networks},  
  author={Xu Chen, Jiangchao Yao, Maosen Li, Ya Zhang, Yanfeng Wang},  
  booktitle={ACM Transactions on the Web},  
  year={2019}  
}  
