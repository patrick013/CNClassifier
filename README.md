# Text classification of clinical notes 
----
Clinical notes usually contain crucial medical information of patients, such as medications, history, physical examinations, and so on. This project aims to classify short clinical notes to its corresponding category by using Navie Bayes algorithm and SVM. The default categories are **Medications**, **Hospital Course**, **Laboratories**, **Physical Examinations** and **History**.

This is a sample of Physical Examination:
> Physical examination revealed temperature was 96.9 , heart rate was 121 , blood pressure was 122/86 , respiratory rate was 22 , and oxygen saturation was 96% on room air . In general , the patient looked acutely and chronically ill . Head , eyes , ears , nose , and throat examination revealed the oropharynx was clear . Sclerae were anicteric . Mucous membranes were moist . Cardiovascular examination revealed tachycardic first heart sound and second heart sound . No murmurs , rubs , or gallops . Lungs revealed decreased breath sounds and dullness to percussion in the left lung base . The abdomen was distended and firm . Positive bowel sounds . Extremity examination revealed no clubbing , cyanosis , or edema .

And the following is an example of Medication paragraph: 
>Atrovent . Aspirin . Flonase . Quinine Celebrex .Compazine . Oxybutynin Amitriptyline . Zyrtec . Prozac . Trazodone . Humulin 70 units bid . Albuterol . Lasix 40 PO bid . Triamcinolone cream . Miconazole cream Flovent . Nifedipine ER 60 daily . Lisinopril , 30 daily . KCl 10 mEq q day .  Protonix , 40 daily . Lipitor , 20 daily . Methadone 10 bid .

## Usage
Pre-trained model is available. It is very easy to use.
```python
>>> import CNClassifier
>>> note="This is a short note describing a patient's information about history illness"
>>> clsf=CNClassifier.classifier()
>>> print(clsf.letspredict(note))
History
```
You could also choose different features for classifier.
> classifier(datanum, dataset,labels_index,labels_name,algo,feature,tfidf=0) 
- datanum : integer, 1-10, the proportion of the dataset used for training classifier model. For - - instance: datanum=7, 70% dataset would be used in training set
- feature: string, feature="bow", BOW ; feature="skip-gram", Skip-gram; feature="cbow", CBOW;
- tfidf: 0 or 1, when tfidf=1, tfidf would be used while 0 means not use
- dataset: string, the directory of dataset
- labels_index: list, labels or tags for each documents
- labels_nameL list, each label's corresponding category
- algo: string, "mult_nb": multinomial NB, "line_svm": linear SVM

## Training
If you want to train your own dataset:
> classifier_model=classifier(dataset="Your dataset directory",labels_index=['your','labels'],labels_name=['your','category'])


The format of the dataset document should be:
```
    label_index1    This is the first notes of category 1,
    label_index1    This is the second note of category 1,
    label_index2    I am the first one in category 2,
    label_index3    I am in category 3,
```

##Installation
> pip install CNClassifier==1.0.4

## License
```
Copyright [2018-2019] [Wei Ruan]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

