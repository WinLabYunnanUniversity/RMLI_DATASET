# RadioML dtasets with attack interference
As shown in Figure 1, the transmitter sends modulated signals to the receiver through a wireless channel. Since radio waves are invisible, intangible, and propagate in free space, they are extremely sensitive to interference and noise. As a result, in order to mimic received signals that are as accurate as possible, interference and noise must be take into account.
![image](https://github.com/WinLabYunnanUniversity/RMLI_DATASET/blob/master/fig1.png)
# Installation
```
pip install pytorch
pip install torchvision
pip install pickle

```
# Datasets 
Since the existing RadioML datasets take into account the presence of Gaussian noise, both noise and interference affect the classification accuracy in our RadioML2023 datasets.  
[RadioML2016 datasets](https://www.deepsig.ai/datasets)  
 RadioML2023 datasets: `run Gen_interference_datasets.py`

# Experiment result
The experiment takes the size of the input data N=50, and the comparision of the classification accuracy with and without attack interference is summarized in Table 1. It reveals that when attack interference was added to the three datasets, their maximum, minimum, and average recognition accuracies changed in the same trend.

| Dataset types | Avg of RML2016 | Max of RML2016 | Min of RML2016 | Avg of RML2023 | Max of RML2023 | Min of RML2023 |
| ------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| RML2016.10a   | 59.7           | 84.1           | 10.19          | 30.48          | 83.84          | 8.65           |
| RML2016.10b   | 62.06          | 88.54          | 10.93          | 32.04          | 88.36          | 10             |
| 2016.04C.multisnr              |70.53                |   96.89             |15.39                |39.25                |96.89                |2.54                |


