# PTG-PLM
Implementation of PTG-PLM: Predicting Post-Translational Glycosylation and Glycation Sites Using Protein Language Models and Deep Learning. PTG-PLM is a model for PTM glycosylation and glycation site prediction. However, it also provides customized model training that enables users to train and predict other PTM prediction models by adjusting the parameters of the training and prediction processes such as: datasets, PTM site residues, and window size.
## Environment Setup
All the implementation done using Python 3.7.13 on Google Colab Pro (https://colab.research.google.com) with GPUs (RAM 16g) and high RAM (28G). Used packages installation can be done by:
``` 
pip install -r requirements.txt
```
### Dataset Format
There are two files that should be prepared for training and prediction:
* Protein sequences FASTA file sample:
```
>P07998
MALEKSLVRLLLLVLILLVLGWVQPSLGKESRAKKFQRQHMDSDSSPSSSSTYCNQMMRRRNMTQGRCKPVNTFVHEPLVDVQNVCFQEKVTCKNGQGNCYKSNSSMHITDCRLTNGSRYPNCAYRTSPKERHIIVACEGSPYVPVHFDASVEDST
>P78380
MTFDDLKIQTVKDQPDEKSNGKKAKGLQFLYSPWWCLAAATLGVLCLGLVVTIMVLGMQLSQVSDLLTQEQANLTHQKKKLEGQISARQQAEEASQESENELKEMIETLARKLNEKSKEQMELHHQNLNLQETLKRVANCSAPCPQDWIWHGENCYLFSSGSFNWEKSQEKCLSLDAKLLKINSTADLDFIQQAISYSSFPFWMGLSRRNPSYPWLWEDGSPLMPHLFRVRGAVSQTYPSGTCAYIQRGAVYAENCILAAFSICQKKANLRAQ
....
```
* Positive sites CSV file sample:
```
ID,Position
P07998,62
P07998,104
P07998,116
P78380,139
P56373,139
P56373,170
P56373,194
```
The first column "ID" represent the protein name or ID and the secod column "Position" represent the PTM positive site.
#### 
