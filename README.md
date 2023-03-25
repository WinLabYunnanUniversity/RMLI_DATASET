## 1. Get datasets
RadioML datasets can be obtained in the following：

 [link] (https://www.deepsig.ai/datasets)

## 2. Requirements
- pytorch
- pickle
- matplotlib
- numpy 
## 3. How to use it
Just download the repository.
### 3.1 Run with ConvNet classification model
```python 
python Gen_interference_dataset.py
```
### 3.2 Run with other classification models
1. Use your classification model to train the data and get the classification model parameters.

2. With your model and parameters, replace the following in the Gen_interference_dataset.py：
```python
   model = ConvNet128(len(all_data[-1])).to(device=device)  
   model.load_state_dict(torch.load('./data/saved_Convnetmodel(rml2016c).pth', map_location=device))  
   model.eval()
```

