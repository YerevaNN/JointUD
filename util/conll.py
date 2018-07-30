import os


def feat_to_col(features):
    columns = {}
    all_features = [
        'abbr', 'animacy', 'aspect', 'case', 'definite', 'degree', 'evident', 
        'foreign', 'gender', 'mood', 'numtype', 'number', 'person', 'polarity',
        'polite', 'poss', 'prontype', 'reflex', 'tense', 'verbform', 'voice'
    ]

    for feature in features.strip().split('|'):
        f = feature.strip().lower()
        if '=' in f:
            feat, _, val = f.partition('=')
        else:
            feat = f
            val = '_'
        columns.update({feat: val})

    return list(map(lambda x: columns[x] if x in columns.keys() else '_', all_features))


def conll_write(output_path, sentences, headers):
    dirname = os.path.dirname(output_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(output_path, 'w') as file:
        for sentence in sentences:
            file.write('#')
            file.write('\t'.join(headers))
            file.write('\n')
            for token in range(len(sentence[headers[0]])):
                data = [sentence[key][token] for key in headers]
                file.write('\t'.join(data))
                file.write('\n')
            file.write('\n')


def conll_read(input_path, cols, comment_symbol=None, val_transformation=None):
    sentences = []
    sent_tpl = {name: [] for name in cols.values()}
    sentence = {name: [] for name in sent_tpl.keys()}
    new_data = False
    
    with open(input_path, encoding='utf-8') as file:
        for line in file.readlines():
            l = line.strip()
            if len(l) == 0 or (comment_symbol is not None and l.startswith(comment_symbol)):
                if new_data:      
                    sentences.append(sentence)
                    sentence = {name: [] for name in sent_tpl.keys()}
                    new_data = False
                continue
            
            splits = l.split()
            if len(splits) > 6:
                splits = splits[:5] + feat_to_col(splits[5]) + splits[6:]
                for col_idx, col_name in cols.items():
                    val = splits[col_idx]
                    if val_transformation is not None:
                        val = val_transformation(col_name, val, splits)
                    sentence[col_name].append(val)  
                new_data = True  
    
    if new_data:
        sentences.append(sentence)
    
    return sentences
