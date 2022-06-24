# iBERT
code for iBERT
---
For regression tasks(STS-B, SICK-STS), train the adaptor:
```
python run_reg_train.py --dataset_name 'STS-B' --epochs_num 10 --kg_layer '[[4,10,11], [3,4,5], [2,3,4]]'
```
Then test the model:
```
python run_reg_test.py --dataset_name 'STS-B' --kg_layer '[[4,10,11], [3,4,5], [2,3,4]]'
```


For classification tasks(TwitterURL, QQP, SICK-NLI, SciTail), train the adaptor:
```
python run_class_train.py --dataset_name 'TwitterURL' --epochs_num 20 --kg_layer '[[9,11], [8,9], [2,9]]'
```
Then test the model:
```
python run_class_test.py --dataset_name 'TwitterURL' --kg_layer '[[9,11], [8,9], [2,9]]'
```

kg_layer is the hyperparameters of enhancement layers. kg_layer[0] represents the layers of local
enhancement with semantics, kg_layer[1] represents the layers of local enhancement
with topic and kg_layer[2] represents the layers of global enhancement.

Please download pretrained_model [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip), and put it in 'models/' folder.<br>
Please download the data we have processed [here](https://drive.google.com/file/d/1Jz0IV9EEQ5o8ny_Fw53Qjy0OlzgIEGli/view?usp=sharing!)

