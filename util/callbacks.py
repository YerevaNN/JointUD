import os
import json
import math
from random import shuffle
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

from keras import callbacks
from keras import utils


class TensorBoard(callbacks.TensorBoard):
    def __init__(self, log_dir='./logs',
                 prefixes=['val_'], **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.log_dirs = {
            prefix: os.path.join(log_dir, prefix)
            for prefix in prefixes
        }

    def set_model(self, model):
        # Setup writer for validation metrics
        self.writers = {
            prefix: tf.summary.FileWriter(log_dir)
            for prefix, log_dir in self.log_dirs.items()
        }
        super(TensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics.
        logs = logs or {}
        epoch = logs.get('subepoch', epoch)

        for prefix, writer in self.writers.items():
            for name, value in logs.items():
                if not name.startswith(prefix):
                    continue
                # print(name, value, type(value))
                name = name.replace(prefix, '')
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = name
                writer.add_summary(summary, epoch) #TODO speed
            logs = {k: v for k, v in logs.items() if not k.startswith(prefix)}
            writer.flush()

        super(TensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TensorBoard, self).on_train_end(logs)
        for prefix, writer in self.writers.items():
            writer.close()

class Metrics(callbacks.Callback):
    def __init__(self, datas, labelKeys, model_name='',
                 validation_rate=1, *args, **kwargs):
        self.datas = datas
        self.labelKeys = labelKeys
        self.validation_rate = validation_rate
        self.model_name = model_name
        super(Metrics, self).__init__(*args, **kwargs)
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        epoch *= self.validation_rate
        logs['subepoch'] = np.array(epoch)
        for name, gen, targ_gen, steps in self.datas:

            input = []
            for v in range(steps):
                inp = gen.__next__()
                input.append(inp)
            
            target = []
            for v in range(steps):
                tgt = targ_gen.__next__()
                target.append(tgt)

            prediction = []
            for i in input:
                pred = self.model.predict(i, verbose=False)
                prediction.append([p.argmax(axis=-1) for p in pred])

            # Evaluation
            # Lemma
            lemma_count = lemma_corr = 0
            for sent in range(len(target)):
                for token in range(len(target[sent][0])):
                    lemma_targ = target[sent][0][token]
                    lemma_pred = np.expand_dims(prediction[sent][0][token], axis=-1)

                    for i in range(lemma_pred.shape[0]):
                        lemma_count += 1
                        if np.array_equal(lemma_targ[i], lemma_pred[i]):
                            lemma_corr += 1
            
            acc = lemma_corr / lemma_count

            print('Lemma Accuracy: {}'.format(acc))
            logs[name + 'lemma_acc'] = acc

            # Labels
            val_targ, val_predict = {}, {}

            for sent in range(len(target)):
                for label in range(len(self.labelKeys)):
                    val_targ.setdefault(label, [])
                    val_predict.setdefault(label, [])
                    for token in range(len(target[sent][label+1])):
                        val_targ[label] += target[sent][label+1][token].squeeze(axis=-1).tolist()
                        val_predict[label] += prediction[sent][label+1][token].tolist()

            F1_labels = []
            R_Labels = []
            P_Labels = []

            for p in range(len(val_predict)):
                F1 = f1_score(val_targ[p], val_predict[p], average='macro')
                R = recall_score(val_targ[p], val_predict[p], average='macro')
                P = precision_score(val_targ[p], val_predict[p], average='macro')

                F1_labels.append(F1)
                R_Labels.append(R)
                P_Labels.append(P)

                print('Label {} - F1: {}, P: {}, R: {}'.format(p, F1, R, P))

                logs[name + str(p) + '_F1'] = F1

            F1 = sum(F1_labels) / len(F1_labels)
            R = sum(R_Labels) / len(R_Labels)
            P = sum(P_Labels) / len(P_Labels)

            self.val_f1s.append(F1)
            self.val_recalls.append(R)
            self.val_precisions.append(P)

            print('Model - F1: {}, P: {}, R: {}'.format(F1, R, P))

            logs['fscore'] = F1
            logs['precision'] = P
            logs['recall'] = R
