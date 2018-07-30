import os.path
import gzip
import pickle as pkl
import logging
import nltk
from nltk import FreqDist
import numpy as np

from util.embeddings import word_normalize
from util.conll import conll_read


def prepare_dataset(embeddings_path,
                    dataset_file,
                    char_path=None,
                    threshold_unknown_tokens=50,
                    comment_symbol=None):
    emd_name = os.path.splitext(embeddings_path)[0]
    ds_name = dataset_file[0]
    output_path = 'pkl/{}.pkl'.format(ds_name)

    if os.path.isfile(output_path):
        logging.info('Using existent pickle file')
        return output_path

    if not os.path.isfile(embeddings_path):
        logging.info('Embeddings file was not found')
        exit()

    logging.info('Generate new embeddings files for a dataset: {}'.format(output_path))

    # Read in word embeddings
    logging.info('Read file: {}'.format(embeddings_path))
    word_to_idx = {}
    needed_vocab = {}
    embeddings = []
    emd_dim = None
    error_lines = 0

    if embeddings_path.endswith('.gz'):
        emd_file = gzip.open(embeddings_path, 'rt')
    else:
        emd_file = open(embeddings_path, 'rb')

    for l in emd_file:
        try:
            line = l.decode('utf8')
        except:
            error_lines += 1
            continue

        split = line.rstrip().split(' ')
        word = split[0]

        if emd_dim is None:
            emd_dim = len(split)-1

        if len(split)-1 != emd_dim:
            logging.info(('ERROR: A line in the embeddings file had more '
                          + 'or less  dimensions than expected. Skip token.'))
            continue

        if len(word_to_idx) == 0:
            word_to_idx["PADDING_TOKEN"] = len(word_to_idx)
            vector = np.zeros(emd_dim)
            embeddings.append(vector)

            word_to_idx["UNKNOWN_TOKEN"] = len(word_to_idx)
            vector = np.random.uniform(-0.25, 0.25, emd_dim)
            embeddings.append(vector)

        vector = np.array([float(num) for num in split[1:]])

        if len(needed_vocab) == 0 or word in needed_vocab:
            if word not in word_to_idx:
                embeddings.append(vector)
                word_to_idx[word] = len(word_to_idx)

    logging.info('Failed to decode {} lines'.format(error_lines))

    if threshold_unknown_tokens >= 0:
        cols_idx = {y: x for x, y in dataset_file[1].items()}
        token_idx = cols_idx['tokens']
        dataset_path = 'data/{}/train.txt'.format(dataset_file[0])

        fd = extend_embeddings(dataset_path, token_idx, word_to_idx)

        added = 0
        for word, freq in fd.most_common(10000):
            if freq < threshold_unknown_tokens:
                break

            added += 1
            word_to_idx[word] = len(word_to_idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)
            embeddings.append(vector)

            assert len(word_to_idx) == len(embeddings)

        logging.info('Added words: {}'.format(added))

    embeddings = np.array(embeddings)

    pkl_obj = {
        'embeddings': embeddings,
        'word_to_idx': word_to_idx,
        'dataset': None,
    }

    casing_to_idx = get_casing_vocab()
    name, cols = dataset_file
    train_data = 'data/{}/train.txt'.format(name)
    dev_data = 'data/{}/dev.txt'.format(name)
    test_data = 'data/{}/test.txt'.format(name)
    paths = [train_data, dev_data, test_data]

    pkl_obj['dataset'] = create_pkl(paths, word_to_idx,
                                    casing_to_idx, cols,
                                    comment_symbol,
                                    char_path=char_path)

    file = open(output_path, 'wb')
    pkl.dump(pkl_obj, file, -1)
    file.close()

    logging.info('DONE. Embeddings file saved: {}'.format(output_path))

    return output_path


def extend_embeddings(filename, token_idx, word_to_idx):
    """ Extend embeddings file with new tokens """
    fd = nltk.FreqDist()
    with open(filename, 'rb') as file:
        for l in file:
            line = l.decode('utf-8', 'ignore')

            if line.startswith('#'):
                continue

            splits = line.strip().split()

            if len(splits) > 1:
                word = splits[token_idx]
                word_normalized = word_normalize(word.lower())

                if (word not in word_to_idx and
                    word.lower() not in word_to_idx and
                    word_normalized not in word_to_idx):
                    fd[word_normalized] += 1
    return fd


def add_char_info(sentences):
    """ Breaks every token into characters """
    for sent_idx in range(len(sentences)):
        sentences[sent_idx]['characters'] = []
        for token_idx in range(len(sentences[sent_idx]['tokens'])):
            token = sentences[sent_idx]['tokens'][token_idx]
            chars = [c for c in token]
            sentences[sent_idx]['characters'].append(chars)
    return sentences


def add_lemma_info(sentences):
    """ Breaks every lemma into characters """
    for sent_idx in range(len(sentences)):
        sentences[sent_idx]['lemma_characters'] = []
        for token_idx in range(len(sentences[sent_idx]['lemma'])):
            token = sentences[sent_idx]['lemma'][token_idx]
            chars = ['\t'] + [c for c in token] + ['\n']
            sentences[sent_idx]['lemma_characters'].append(chars)
    return sentences


def add_casing_info(sentences):
    """ Adds casing information """
    for sent_idx in range(len(sentences)):
        sentences[sent_idx]['casing'] = []
        for token_idx in range(len(sentences[sent_idx]['tokens'])):
            token = sentences[sent_idx]['tokens'][token_idx]
            sentences[sent_idx]['casing'].append(get_casing(token))
    return sentences
       
       
def get_casing(word):
    casing = 'other'
    
    digits_len = 0
    for char in word:
        if char.isdigit():
            digits_len += 1
            
    digit_fraction = digits_len / float(len(word))
    
    if word.isdigit():
        casing = 'numeric'
    elif digit_fraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():
        casing = 'allLower'
    elif word.isupper():
        casing = 'allUpper'
    elif word[0].isupper():
        casing = 'initialUpper'
    elif digits_len > 0:
        casing = 'contains_digit'
    
    return casing


def get_casing_vocab():
    entries = [
        'PADDING', 'other', 'numeric', 'mainly_numeric',
        'allLower', 'allUpper', 'initialUpper', 'contains_digit',
    ]
    return {entries[idx]: idx for idx in range(len(entries))}


def create_matrices(sentences, mappings):
    data = []
    tokens_len = 0
    unknown_tok_len = 0    
    missing_tokens = FreqDist()
    padded_sents = 0

    for s in sentences:
        row = {name: [] for name in list(mappings.keys())+['raw_tokens']}
        
        for mapping, str_to_idx in mappings.items():    
            if mapping not in s:
                continue
                    
            for entry in s[mapping]:                
                if mapping.lower() == 'tokens':
                    tokens_len += 1
                    idx = str_to_idx['UNKNOWN_TOKEN']
                    if entry in str_to_idx:
                        idx = str_to_idx[entry]
                    elif entry.lower() in str_to_idx:
                        idx = str_to_idx[entry.lower()]
                    elif word_normalize(entry) in str_to_idx:
                        idx = str_to_idx[word_normalize(entry)]
                    else:
                        unknown_tok_len += 1    
                        missing_tokens[word_normalize(entry)] += 1
                    row['raw_tokens'].append(entry)

                elif mapping.lower() == 'characters':  
                    idx = []
                    for c in entry:
                        if c in str_to_idx:
                            idx.append(str_to_idx[c])
                        else:
                            idx.append(str_to_idx['UNKNOWN']) 

                elif mapping.lower() == 'lemma_characters':  
                    idx = []
                    for c in entry:
                        if c in str_to_idx:
                            idx.append(str_to_idx[c])
                        else:
                            idx.append(str_to_idx['UNKNOWN'])                           
                                      
                else:
                    idx = str_to_idx[entry]
                                    
                row[mapping].append(idx)
                
        if len(row['tokens']) == 1:
            padded_sents += 1
            for mapping, str_to_idx in mappings.items():
                if mapping.lower() == 'tokens':
                    row['tokens'].append(mappings['tokens']['PADDING_TOKEN'])
                    row['raw_tokens'].append('PADDING_TOKEN')
                elif mapping.lower() == 'characters':
                    row['characters'].append([0])
                elif mapping.lower() == 'lemma_characters':
                    row['lemma_characters'].append([0])
                else:
                    row[mapping].append(0)

        data.append(row)
    
    if tokens_len > 0:
        unknown_tok_pc = unknown_tok_len / float(tokens_len) * 100
        logging.info('Unknown tokens: {}'.format(unknown_tok_pc))
        
    return data


def create_pkl(datasets, word_to_idx, casing_to_idx, cols,
               comment_symbol=None, val_transformation=None, char_path=None):

    train = conll_read(datasets[0], cols, comment_symbol, val_transformation)
    dev = conll_read(datasets[1], cols, comment_symbol, val_transformation)
    test = conll_read(datasets[2], cols, comment_symbol, val_transformation)
   
    mappings = create_mappings(train + dev + test)
    mappings['tokens'] = word_to_idx
    mappings['casing'] = casing_to_idx

    if char_path and os.path.isfile(char_path):
        with open(char_path, 'r', encoding='utf-8') as f:
            alphabet = f.readlines()[0]
    else:
        logging.info('Characters path is invalid')
        alphabet = 'qwertyuiopasdfghjklzxcvbnm'
        alphabet += alphabet.upper()

    charset = {
        'PADDING': 0,
        'UNKNOWN': 1,
        '\t': 2,
        '\n': 3,
    }
    charset_letters = ''.join([' 0123456789',
                               '.,-_()[]{}!?:;#\'\"/\\%$`&=*+@^~|',
                               alphabet,
                               ])

    for c in charset_letters:
        charset[c] = len(charset)

    mappings['characters'] = charset
    mappings['lemma_characters'] = charset
    
    add_char_info(train)
    add_lemma_info(train)
    add_casing_info(train)
    
    add_char_info(dev)
    add_lemma_info(dev)
    add_casing_info(dev)
    
    add_char_info(test)
    add_lemma_info(test)
    add_casing_info(test)
    
    train_matrix = create_matrices(train, mappings)
    dev_matrix = create_matrices(dev, mappings)
    test_matrix = create_matrices(test, mappings)
    
    data = {'mappings': mappings,
            'train': train_matrix,
            'dev': dev_matrix,
            'test': test_matrix,
            }

    return data


def create_mappings(sentences):
    sentence_keys = list(sentences[0].keys())
    sentence_keys.remove('tokens')

    if 'lemma' in sentence_keys:
        sentence_keys.remove('lemma')
    
    vocabs = {name: {'O': 0} for name in sentence_keys}
    
    for sentence in sentences:
        for name in sentence_keys:
            for item in sentence[name]:              
                if item not in vocabs[name]:
                    vocabs[name][item] = len(vocabs[name]) 
    return vocabs


def load_dataset(pickle_path):
    f = open(pickle_path, 'rb')
    pkl_obj = pkl.load(f)
    f.close()
    
    return pkl_obj['embeddings'], pkl_obj['word_to_idx'], pkl_obj['dataset']


def prepare_predict_data(data, char_len, lstm_units):
    def pad():
        for sent_idx in range(len(data)):
            for token_idx in range(len(data[sent_idx]['characters'])):
                token = data[sent_idx]['characters'][token_idx]

                if len(token) < char_len:
                    data[sent_idx]['characters'][token_idx] = np.pad(
                        token,
                        (0, char_len-len(token)),
                        'constant')
                elif len(token) > char_len:
                    data[sent_idx]['characters'][token_idx] = token[:char_len]

            for token_idx in range(len(data[sent_idx]['lemma_characters'])):
                token = data[sent_idx]['lemma_characters'][token_idx]

                if len(token) < char_len:
                    data[sent_idx]['lemma_characters'][token_idx] = np.pad(
                        token,
                        (0, char_len-len(token)),
                        'constant')
                elif len(token) > char_len:
                    data[sent_idx]['lemma_characters'][token_idx] = token[:char_len-1] + [token[-1]]

    def positional():
        for idx in range(len(data)):
            t_len = len(data[idx]['characters'])
            char_pos = np.zeros(shape=(t_len, char_len))

            i = 0
            for c in range(len(data[idx]['characters'])):
                c_np = np.asarray(data[idx]['characters'][c])
                positional = np.arange(1, char_len+1)[c_np > 0][::-1]

                vec = np.zeros(char_len)
                vec[:len(positional)] = positional

                char_pos[i] = vec
                i += 1
            data[idx]['positionalEmd'] = char_pos

    def lmtz():
        bin_dim = lstm_units

        for idx in range(len(data)):
            input_decoder, state_l = [], []

            for _ in data[idx]['tokens']:
                input_decoder.append(np.zeros((char_len, bin_dim)))
                state = np.zeros(bin_dim)
                state[2] += 1
                state_l.append(state)

            data[idx]['lmtz_inp'] = input_decoder
            data[idx]['lmtz_state'] = state_l

    pad()
    positional()
    lmtz()
    
    return data


def get_key(d, val):
    for k, v in d.items():
        if v == val:
            return k


def parse_lemma(word, mappings):
    readable_word = ''
    for l in word:
        if l == 0 or l == 3:
            break

        if l == 2:
            continue

        readable_word += get_key(mappings['lemma_characters'], l)

    return readable_word


def parse_feat(f, features):
    def get_featname(n):
        labels = {
            'prontype': 'PronType',
            'numtype': 'NumType',
            'poss': 'Poss',
            'reflex': 'Reflex',
            'foreign': 'Foreign',
            'abbr': 'Abbr',
            'gender': 'Gender',
            'animacy': 'Animacy',
            'number': 'Number',
            'case': 'Case',
            'definite': 'Definite',
            'degree': 'Degree',
            'verbform': 'VerbForm',
            'mood': 'Mood',
            'tense': 'Tense',
            'aspect': 'Aspect',
            'voice': 'Voice',
            'evident': 'Evident',
            'polarity': 'Polarity',
            'person': 'Person',
            'polite': 'Polite',
        }

        return labels[n.lower()]

    f = '|'.join([
        '{}={}'.format(get_featname(i), j.capitalize())
        for i, j in list(zip(features, f)) if j and j != '_'
    ])

    if f.strip():
        return f

    return '_'
