from collections import defaultdict

import torch
from fastNLP import DataSetIter, SequentialSampler
from fastNLP.core.utils import _build_args, _move_dict_value_to_device, _get_model_device


def predict(model, data, seq_len_field_name, batch_size):
    """ Given a trained model and a dataset, conducts the prediction """
    prev_training = model.training
    model.eval()
    device = _get_model_device(model)
    batch_output = defaultdict(list)
    data_iterator = DataSetIter(data, batch_size=batch_size, sampler=SequentialSampler(), as_numpy=False)

    with torch.no_grad():
        for batch_x, _ in data_iterator:
            _move_dict_value_to_device(batch_x, _, device=device)
            refined_batch_x = _build_args(model.predict, **batch_x)
            prediction = model.predict(**refined_batch_x)
            
            if seq_len_field_name is not None:
                seq_lens = batch_x[seq_len_field_name].tolist()

            for key, value in prediction.items():
                value = value.cpu().numpy()
                if len(value.shape) == 1 or (len(value.shape) == 2 and value.shape[1] == 1):
                    batch_output[key].extend(value.tolist())
                else:
                    if seq_len_field_name is not None:
                        tmp_batch = []
                        for idx, seq_len in enumerate(seq_lens):
                            tmp_batch.append(value[idx, :seq_len])
                        batch_output[key].extend(tmp_batch)
                    else:
                        batch_output[key].append(value)

    model.train(prev_training)

    return batch_output


def write_predictions(model, data, seq_len_field_name, batch_size, 
                      word_vocab, target_vocab, output_dir):
    """ Write predictions to file, needs vocab """
    batch_output = predict(model, data, seq_len_field_name, batch_size)
    unified_preds = batch_output['pred']

    tag_seqs = []
    for i in range(len(unified_preds)):
        tag_seq = [target_vocab.to_word(w) for w in unified_preds[i].tolist()]
        tag_seqs.append(tag_seq)

    lines = []
    for idx, qa in enumerate(data):
        tokenized_q = [word_vocab.to_word(w) for w in qa['question']]
        tokenized_a = [word_vocab.to_word(w) for w in qa['answer']]
        aspect_seq = tag_seqs[idx]
        assert len(aspect_seq) == len(tokenized_q)
        words = tokenized_q + ['[QA_SEP]'] + tokenized_a
        tags = aspect_seq + ['O']*(len(tokenized_a)+1)
        for i in range(len(words)):
            lines.append(f"{words[i]}\t{tags[i]}\n")
        lines.append('\n')

    with open(output_dir, 'w+') as f:
        f.writelines(lines)

    print(f'Done, data is written to {output_dir}')
