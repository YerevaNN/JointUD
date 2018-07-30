import nltk
import pickle
import numpy as np
import argparse

from keras.models import load_model
from keras import backend as K

from layers.time_distributed import TimeDistributed
from layers.recurrent_cell import RecurrentCell
from layers.repeat_3d import Repeat3DVector
from data import add_char_info, create_matrices
from data import add_casing_info, prepare_predict_data
from data import parse_lemma, parse_feat, get_key

K.set_floatx('float64')

np.random.seed(10)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='', help='Model path', type=str)
parser.add_argument('--mapping_path', default='', help='Mapping path', type=str)
parser.add_argument('--input_path', default='', help='Input path', type=str)
parser.add_argument('--output_path', default='output.txt', help='Output path', type=str)

args = parser.parse_args()

print('Preparing dataset..')

with open(args.input_path, 'r', encoding='utf-8') as f:
    text = f.read()

sentences = [{
    'tokens': nltk.word_tokenize(sent)
} for sent in nltk.sent_tokenize(text)]
sentences = add_casing_info(sentences)
sentences = add_char_info(sentences)

output = open(args.mapping_path, 'rb')
info = pickle.load(output)
output.close()

mappings = info['mappings']
char_len = info['char_len']

dataset = create_matrices(sentences, mappings)
dataset = prepare_predict_data(dataset, char_len, lstm_units=160)

nn_input = []
for i in dataset:
    nn_input.append([
        np.asarray([i['tokens']]),
        np.asarray([i['casing']]),
        np.asarray([i['characters']]),
        np.asarray([i['positionalEmd']]),
        np.asarray([i['lmtz_inp']]),
        np.asarray([i['lmtz_state']]),
    ])

print('Loading model..')

model = load_model(
    args.model_path,
    custom_objects={
        'TimeDistributed': TimeDistributed,
        'Repeat3DVector': Repeat3DVector,
        'RecurrentCell': RecurrentCell,
    })

print('Prediction..')

target = []
for i in nn_input:
    pred = model.predict(i)
    output = [p.argmax(axis=-1) for p in pred]

    target.append({})
    for index, val in enumerate(output):
        target[-1].update({index: val[0]})

print('Parsing output..')

parsed = []
for idx in range(len(target)):
    parsed.append({'raw_tokens': dataset[idx]['raw_tokens']})
    for label in target[idx].keys():
        if label == 0:
            lemmas = []
            for i in target[idx][label]:
                lemmas.append(parse_lemma(i, mappings))
            parsed[-1].update({'lemma': lemmas})
        else:
            k, v = list(mappings.items())[label - 1]

            label_val = []
            for i in target[idx][label]:
                label_val.append(get_key(v, i))
            parsed[-1].update({k: label_val})

print('Saving output..')

f = open(args.output_path, 'w', encoding='utf-8')

tpl = '{id}\t{token}\t{lemma}\t{pos}\t_\t{features}\t{head}\t{dep}\t_\t_\n'
features = list(parsed[0].keys())[4:]

for i in parsed:
    idx = 1
    for r in list(zip(*[l for l in list(i.values())])):
        row = tpl.format(
            id=idx,
            token=r[0],
            lemma=r[1],
            pos=r[2],
            features=parse_feat(r[3:], features),
            head='_',
            dep='_')
        f.write(row)
        idx += 1
    f.write('\n')

f.close()
