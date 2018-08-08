# JointUD
#### Multitask neural model for Part-of-Speech tagging, Morphological parsing and Lemmatization
#### Implemented in Keras 2.1.5

## Installation
### Dependencies
 - Keras==2.1.5
 - Theano==0.9.0
 - tensorflow==1.0.0
 - nltk==3.2.2
 - numpy==1.14.2
 - sklearn

### Keras backend
 - Use Theano backend

### Python version
 - python3

## Prerequisite
 - Download pre-trained word vectors to ```wordvec/``` directory.
 - Put ```train.txt```, ```test.txt```, ```dev.txt``` files into ```data/{dataset_name}/``` directory.
 - Write the language alphabet to ```{dataset_name}.chars``` file and save it in ```characters/``` directory.

## Usage
### Training
```
usage: train.py [-h] [--model_name MODEL_NAME] [--nb_epochs NB_EPOCHS]
                [--early_stopping EARLY_STOPPING] [--batch_size BATCH_SIZE]
                [--optimizer OPTIMIZER] [--classifier CLASSIFIER]
                [--loss LOSS] [--dump_prefix DUMP_PREFIX] [--lstm LSTM]
                [--lstm_size LSTM_SIZE] [--dropout DROPOUT]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Model name
  --nb_epochs NB_EPOCHS
                        Number of Epochs
  --early_stopping EARLY_STOPPING
                        Early Stopping
  --batch_size BATCH_SIZE
                        Batch Size
  --optimizer OPTIMIZER
                        Optimizer
  --classifier CLASSIFIER
                        Classifier
  --loss LOSS           Loss
  --dump_prefix DUMP_PREFIX
                        Model dump name prefix
  --lstm LSTM
  --lstm_size LSTM_SIZE
  --dropout DROPOUT
```

### Parsing
```
usage: parse.py [-h] [--model_path MODEL_PATH] [--mapping_path MAPPING_PATH]
                [--input_path INPUT_PATH] [--output_path OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Model path
  --mapping_path MAPPING_PATH
                        Mapping path
  --input_path INPUT_PATH
                        Input path
  --output_path OUTPUT_PATH
                        Output path
```

## Example
 - Input
```
I have been going there since I was a little boy and love the friendly and relaxing atmosphere.
```

 - Output
```
1	I	I	PRON	_	Case=Nom|Number=Sing|Person=1|PronType=Prs	_	_	_	_
2	have	have	VERB	_	Mood=Ind|Tense=Pres|VerbForm=Fin	_	_	_	_
3	been	be	AUX	_	Tense=Past|VerbForm=Part	_	_	_	_
4	going	go	VERB	_	VerbForm=Ger	_	_	_	_
5	there	there	ADV	_	PronType=Dem	_	_	_	_
6	since	since	ADP	_	_	_	_	_	_
7	I	I	PRON	_	Case=Nom|Number=Sing|Person=1|PronType=Prs	_	_	_	_
8	was	be	AUX	_	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	_	_	_	_
9	a	a	DET	_	Definite=Ind|PronType=Art	_	_	_	_
10	little	little	ADJ	_	Degree=Pos	_	_	_	_
11	boy	boy	NOUN	_	Number=Sing	_	_	_	_
12	and	and	CCONJ	_	_	_	_	_	_
13	love	love	VERB	_	VerbForm=Fin	_	_	_	_
14	the	the	DET	_	Definite=Def|PronType=Art	_	_	_	_
15	friendly	friendly	ADJ	_	Degree=Pos	_	_	_	_
16	and	and	CCONJ	_	_	_	_	_	_
17	relaxing	relax	VERB	_	VerbForm=Ger	_	_	_	_
18	atmosphere	atmosphere	NOUN	_	Number=Sing	_	_	_	_
19	.	.	PUNCT	_	_	_	_	_	_
```