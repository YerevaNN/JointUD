import math
import random
import datetime
import logging

from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras import backend as K

from layers.time_distributed import TimeDistributed
from layers.recurrent_cell import RecurrentCell
from layers.repeat_3d import Repeat3DVector
from layers.chain_crf import ChainCRF
from util.callbacks import TensorBoard, Metrics

K.set_floatx('float64')


class MultitaskLSTM:
    def __init__(self, name, embeddings, data, params):
        dataset, labels = data

        self.model_name = name
        self.dataset = dataset
        self.labels =  labels
        self.embeddings = embeddings
        self.lstm_units = params['lstm_units']
        self.char_maxlen = params['char_maxlen']
        self.lmtz = params['lemmatization'] and params['char_embeddings']

        self.token_to_idx = dataset['mappings']['tokens']
        self.casing_to_idx = dataset['mappings']['casing']
        self.char_to_idx = dataset['mappings']['lemma_characters']
        self.idx_to_labels = {labelKey: {v: k for k, v in dataset['mappings'][labelKey].items()}
                              for labelKey in labels
                              }

        self.model = None
        self.early_stopping = params['early_stopping']
        self.batch_size = params['batch_size']

        self.epoch = 0
        self.char_len = 0

        self.addFeatureDimensions = 10
        self.learning_rate_updates = {
            'adam': {
                7: 0.001,
                4: 0.005,
                1: 0.01,
            },
            'rmsprop': {
                7: 0.0005,
                1: 0.001,
            },
            'sgd': {
                5: 0.01,
                3: 0.05,
                1: 0.1,
            }
        }

        self.features = []
        self.sent_len_ranges = {}
        self.mini_batch_ranges = {}
        self.range_len = {}

        self.params = {
            'char_embeddings_size': 30,
            'char_filter_size': 30,
            'char_filter_length': 3,
            'char_lstm_size': 25,
            'task_identifier': False,
            'clipvalue': 0,
            'clipnorm': 1,
        }
        self.params.update(params)

        logging.info('{} train sentences'.format(len(dataset['train'])))
        logging.info('{} dev sentences'.format(len(dataset['dev'])))
        logging.info('{} test sentences'.format(len(dataset['test'])))

        self.build_model()

    def build_model(self):
        tokens_input = Input(shape=(None, ), dtype='int64', name='words_input')
        tokens = Embedding(
            input_dim=self.embeddings.shape[0],
            output_dim=self.embeddings.shape[1],
            weights=[self.embeddings],
            trainable=False)(tokens_input)

        casing_input = Input(shape=(None, ), dtype='int64', name='casing_input')
        case_matrix = np.identity(len(self.casing_to_idx), dtype='float64')
        casing = Embedding(
            input_dim=case_matrix.shape[0],
            output_dim=case_matrix.shape[1],
            weights=[case_matrix],
            trainable=False)(casing_input)

        self.features = ['tokens', 'casing']
        input_layers = [tokens, casing]
        input_nodes = [tokens_input, casing_input]

        # Character Embeddings
        if self.params['char_embeddings']:
            logging.info('Pad characters to uniform length')
            self.pad()
            char_len = self.char_len
            logging.info('Words padded to {} characters'.format(char_len))

            charset = self.dataset['mappings']['characters']

            char_embeddings_size = self.params['char_embeddings_size']
            
            char_embeddings = []
            for _ in charset:
                limit = math.sqrt(3.0 / char_embeddings_size)
                vector = np.random.uniform(-limit, limit, char_embeddings_size)
                char_embeddings.append(vector)

            char_embeddings[0] = np.zeros(char_embeddings_size)
            char_embeddings = np.asarray(char_embeddings)

            chars_input = Input(shape=(None, char_len), dtype='int64', name='char_input')
            chars_emd = TimeDistributed(Embedding(
                input_dim=char_embeddings.shape[0],
                output_dim=char_embeddings.shape[1],
                weights=[char_embeddings],
                trainable=True,
                mask_zero=True),
                name='char_emd')(chars_input)

            chars = chars_emd

            if self.params['char_embeddings'].lower() == 'lstm':
                chars = TimeDistributed(Bidirectional(LSTM(
                    self.params['char_lstm_size'],
                    return_sequences=False)),
                    name='char_lstm')(chars)
            else:
                chars = TimeDistributed(Convolution1D(
                    self.params['char_filter_size'],
                    self.params['char_filter_length'],
                    border_mode='same'),
                    name='char_cnn')(chars)
                chars = TimeDistributed(GlobalMaxPooling1D(), name='char_pooling')(chars)

            input_layers.append(chars)
            input_nodes.append(chars_input)
            self.features.append('characters')

        shared_layer = Concatenate(axis=-1)(input_layers)

        # Add LSTMs
        cnt = 1
        for size in self.params['lstm_size']:
            shared_layer = LSTM(
                size,
                return_sequences=True,
                name='shared_LSTM_{cnt}'.format(cnt=cnt))(shared_layer)

            dropout = self.params['dropout']
            if dropout > 0:
                shared_layer = TimeDistributed(Dropout(dropout),
                    name='shared_dropout_{d}_{c}'.format(d=dropout, c=cnt))(shared_layer)

            cnt += 1

        outputs = []

        # Lemmatization
        if self.lmtz:
            lmtz_positional_input = Input(
                shape=(None, char_len),
                dtype='int64',
                name='lemma_position_input')
            lmtz_positional_emd = TimeDistributed(Embedding(
                input_dim=char_len+1, output_dim=5, trainable=True),
                name='char_position_emd')(lmtz_positional_input)

            lmtz_decoder_input = Input(
                shape=(None, char_len, self.lstm_units),
                dtype='int64',
                name='lemma_char_input')

            lmtz_lstm_input = Repeat3DVector(char_len)(shared_layer)

            decoder_merged_input = Concatenate(axis=-1)([
                lmtz_lstm_input,
                chars_emd,
                lmtz_positional_emd,
                lmtz_decoder_input,
                ])

            lmtz_decoder_state_l = Input(
                shape=(None, self.lstm_units),
                dtype='float64',
                name='lemma_char_state')

            lmtz_gru_cell = RecurrentCell(size, dense_units=size)
            lmtz_gru = RNN(lmtz_gru_cell, return_sequences=True)

            lmtz_Y = TimeDistributed(lmtz_gru, name='lemma_gru_decoder')([
                decoder_merged_input,
                shared_layer,
                lmtz_decoder_state_l,
                ])

            loss_func = 'sparse_categorical_crossentropy'

            input_nodes.append(lmtz_positional_input)
            input_nodes.append(lmtz_decoder_input)
            input_nodes.append(lmtz_decoder_state_l)

            outputs.append((lmtz_Y, loss_func))

        # POS and Morph. features
        loss_weights = {}

        for label in self.labels:
            output = shared_layer
            decoder = self.params['classifier']

            if decoder == 'softmax':
                layer_name = '{}_softmax'.format(label)
                output = Dense(
                    len(self.dataset['mappings'][label]),
                    activation='softmax',
                    name=layer_name)(output)
                loss_func = 'sparse_categorical_crossentropy'

            elif decoder == 'crf':
                layer_name = '{}_hidden_lin_layer'.format(label)
                output = Dense(
                    len(self.dataset['mappings'][label]),
                    activation=None,
                    name=layer_name)(output)
                crf = ChainCRF(name='{}_CRF'.format(label))
                output = crf(output)
                loss_func = crf.sparse_loss

            elif decoder == 'tanh-crf':
                layer_name = '{}_hidden_tanh_layer'.format(label)
                output = Dense(
                    len(self.dataset['mappings'][label]),
                    activation='tanh',
                    name=layer_name)(output)
                crf = ChainCRF()
                output = crf(output)
                loss_func = crf.sparse_loss

            else:
                assert False

            if label.lower() == 'pos':
                loss_weights.update({layer_name: 0.2})

            outputs.append((output, loss_func))

        opt_params = {}

        if self.params['clipnorm'] > 0:
            opt_params['clipnorm'] = self.params['clipnorm']

        if self.params['clipvalue'] > 0:
            opt_params['clipvalue'] = self.params['clipvalue']

        if self.params['optimizer'].lower() == 'adam':
            opt = Adam(**opt_params)
        elif self.params['optimizer'].lower() == 'nadam':
            opt = Nadam(**opt_params)
        elif self.params['optimizer'].lower() == 'rmsprop':
            opt = RMSprop(**opt_params)
        elif self.params['optimizer'].lower() == 'adadelta':
            opt = Adadelta(**opt_params)
        elif self.params['optimizer'].lower() == 'adagrad':
            opt = Adagrad(**opt_params)
        elif self.params['optimizer'].lower() == 'sgd':
            opt = SGD(lr=0.1, **opt_params)

        model = Model(inputs=input_nodes, outputs=[o for o, _ in outputs])

        model.compile(
            loss=[l for _, l in outputs],
            optimizer=opt,
            loss_weights=loss_weights)

        model.summary(line_length=200)
        logging.info(model.get_config())
        logging.info('Optimizer: {} - {}'.format(
            type(model.optimizer),
            model.optimizer.get_config()))

        self.model = model

    def train(self, epochs):
        name = self.model_name

        self.prepare_dataset(dataset_name='train')
        self.prepare_dataset(dataset_name='dev')

        dt = datetime.datetime.now().isoformat()

        tensor_board = TensorBoard(
            log_dir='./logs/{}/{}'.format(name, dt),
            write_graph=True,
            write_images=True)

        def lr_schedule(epoch=0, learning_rate=0):
            lr = 0.01
            for e, lr in self.learning_rate_updates[self.params['optimizer']].items():
                if epoch >= e:
                    break
            return lr

        learning_rate = LearningRateScheduler(lr_schedule)

        early_stopping = EarlyStopping(min_delta=0.00001, patience=self.early_stopping)

        metrics = Metrics(
            [(
                'val_',
                self.minibatch_iterate_dev_dataset(return_value='input'),
                self.minibatch_iterate_dev_dataset(return_value='target'),
                self.range_len['dev'],
            )],
            self.labels,
            model_name=name)

        checkpoint = ModelCheckpoint((
                self.params['model_path']
                + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
            ),
            monitor='val_loss',
            save_best_only=True)

        self.model.fit_generator(
            self.minibatch_iterate_train_dataset(),
            steps_per_epoch=self.range_len['train'],
            epochs=epochs,
            validation_data=self.minibatch_iterate_dev_dataset(
                return_value='input,target'),
            validation_steps=self.range_len['dev'],
            callbacks=[
                learning_rate, early_stopping, checkpoint, metrics, tensor_board
            ])

    def prepare_dataset(self, dataset_name='train'):
        # Sort train matrix by sentence length
        data = self.dataset[dataset_name]
        data.sort(key=lambda x: len(x['tokens']))

        ranges = []
        prev_sent_len = len(data[0]['tokens'])
        start_idx = 0

        for idx in range(len(data)):
            sent_len = len(data[idx]['tokens'])
            if sent_len != prev_sent_len:
                ranges.append((start_idx, idx))
                start_idx = idx

            prev_sent_len = sent_len

        # Add last sentence
        ranges.append((start_idx, len(data)))

        # Break up ranges into smaller mini batch sizes
        mini_batch_ranges = []
        for batch_range in ranges:
            range_len = batch_range[1] - batch_range[0]

            bins = int(math.ceil(range_len / float(self.batch_size)))
            bin_size = int(math.ceil(range_len / float(bins)))

            for b in range(bins):
                start = b * bin_size + batch_range[0]
                end = min(batch_range[1], (b + 1)*bin_size + batch_range[0])
                mini_batch_ranges.append((start, end))

        self.sent_len_ranges[dataset_name] = ranges
        self.mini_batch_ranges[dataset_name] = mini_batch_ranges

        # Shuffle training data
        if dataset_name == 'train':
            # 1. Shuffle sentences that have the same length
            x = self.dataset['train']
            for data_range in self.sent_len_ranges['train']:
                for i in reversed(range(data_range[0] + 1, data_range[1])):
                    j = random.randint(data_range[0], i)
                    x[i], x[j] = x[j], x[i]

            # 2. Shuffle the order of mini batch ranges
            random.shuffle(self.mini_batch_ranges['train'])

        self.range_len[dataset_name] = len(self.mini_batch_ranges[dataset_name])

    def get_batch(self, dataset_name, idx):
        matrix = self.dataset[dataset_name]
        data_range = self.mini_batch_ranges[dataset_name][
                        idx % len(self.mini_batch_ranges[dataset_name])]

        batches = {
            'labels': [],
            'input': [],
            'lmtz': [],
            'extras': [],
        }

        for l in self.labels:
            pred_labels = np.asarray([
                matrix[idx][l]
                for idx in range(data_range[0], data_range[1])
            ])
            pred_labels = np.expand_dims(pred_labels, -1)
            batches['labels'].append(pred_labels)

        for f in self.features:
            input_data = np.asarray([
                matrix[idx][f]
                for idx in range(data_range[0], data_range[1])
            ])
            batches['input'].append(input_data)

        # Character positional embeddings
        char_pos_inp = []
        for idx in range(data_range[0], data_range[1]):
            t_len = len(matrix[idx]['characters'])
            char_pos = np.zeros(shape=(t_len, self.char_len))
            i = 0
            for c in range(len(matrix[idx]['characters'])):
                c_np = np.asarray(matrix[idx]['characters'][c])
                positional = np.arange(1, self.char_len + 1)[c_np > 0][::-1]

                vec = np.zeros(self.char_len)
                vec[:len(positional)] = positional

                char_pos[i] = vec
                i += 1
            char_pos_inp.append(char_pos)
        batches['extras'].append(np.asarray(char_pos_inp))

        # Lemma batch
        target_decoder, input_decoder, state_l = [], [], []
        bin_dim = self.lstm_units

        for idx in range(data_range[0], data_range[1]):
            target_decoder.append([])
            input_decoder.append([])
            state_l.append([])
            for ch in matrix[idx]['lemma_characters']:
                if dataset_name == 'train':
                    hot = np.eye(bin_dim)[ch]
                    input_decoder[-1].append(hot)
                    state_l[-1].append(np.zeros(bin_dim))
                else:
                    input_decoder[-1].append(
                        np.zeros((self.char_len, bin_dim)))
                    state = np.zeros(bin_dim)
                    state[2] += 1
                    state_l[-1].append(state)

                tgt = np.concatenate((ch[1:], np.zeros((1, ))))
                tgt = [[i] for i in tgt]
                target_decoder[-1].append(tgt)

        batches['lmtz'].append(np.asarray(input_decoder))
        batches['lmtz'].append(np.asarray(target_decoder))
        batches['lmtz'].append(np.asarray(state_l))

        return batches

    def minibatch_iterate_train_dataset(self):
        range_len = self.range_len['train']
        idx = 0

        while True:
            if idx == range_len - 1:
                idx = 0

            batches = self.get_batch(dataset_name='train', idx=idx)

            nn_labels = batches['labels']
            nn_input = batches['input']
            inp, tgt, state = batches['lmtz']
            char_pos = batches['extras'][0]

            idx += 1

            if self.lmtz:
                r_inp = nn_input + [char_pos, inp, state]
                r_tgt = [tgt] + nn_labels
            else:
                r_inp = nn_input
                r_tgt = nn_labels

            yield (r_inp, r_tgt)

    def minibatch_iterate_dev_dataset(self, return_value):
        range_len = self.range_len['dev']
        idx = 0

        while True:
            if idx == range_len - 1:
                idx = 0

            batches = self.get_batch(dataset_name='dev', idx=idx)

            nn_labels = batches['labels']
            nn_input = batches['input']
            inp, tgt, state = batches['lmtz']
            char_pos = batches['extras'][0]

            idx += 1

            if self.lmtz:
                r_input = nn_input + [char_pos, inp, state]
                r_tgt = [tgt] + nn_labels
            else:
                r_input = nn_input
                r_tgt = nn_labels

            if return_value == 'target':
                yield r_tgt
            elif return_value == 'input':
                yield r_input
            else:
                yield (r_input, r_tgt)

    def pad(self):
        max_len = 0
        dataset = self.dataset
        for data in [dataset['train'],
                     dataset['dev'],
                     dataset['test'],
                     ]:
            for sentence in data:
                for token in sentence['characters'] + sentence['lemma_characters']:
                    max_len = max(max_len, len(token))

        if max_len > self.char_maxlen:
            max_len = self.char_maxlen

        for data in [dataset['train'],
                     dataset['dev'],
                     dataset['test'],
                     ]:
            # Pad each other word with zeros
            for s in range(len(data)):
                for t in range(len(data[s]['characters'])):
                    token = data[s]['characters'][t]

                    if len(token) < max_len:
                        data[s]['characters'][t] = np.pad(
                            token,
                            (0, max_len-len(token)),
                            'constant')
                    elif len(token) > max_len:
                        data[s]['characters'][t] = token[:max_len]

                for t in range(len(data[s]['lemma_characters'])):
                    token = data[s]['lemma_characters'][t]

                    if len(token) < max_len:
                        data[s]['lemma_characters'][t] = np.pad(
                            token,
                            (0, max_len-len(token)),
                            'constant')
                    elif len(token) > max_len:
                        data[s]['lemma_characters'][t] = token[:max_len-1] + [token[-1]]

        self.char_len = max_len
