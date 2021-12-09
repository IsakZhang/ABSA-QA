# This script trains & evals the proposed model

import random
import argparse
import os
import numpy as np

import torch
from torch import optim
from fastNLP import SpanFPreRecMetric
from fastNLP import Trainer, Tester, BucketSampler, LossInForward
from fastNLP import EvaluateCallback, GradientClipCallback, WarmupCallback
from fastNLP.embeddings import StaticEmbedding, BertEmbedding

from modules.datapipe import ABSAQAPipe
from models.model import Model
from modules.predict import write_predictions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  


def check_gpu():
    # Check whether or which GPU will be being used
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='elec', choices=['elec', 'bags', 'beauty'])
parser.add_argument('--embed', type=str, default='bert')
parser.add_argument('--bert_dir', type=str, default='data/bert/')
parser.add_argument('--fasttext_dir', type=str, default='data/fasttext/')
parser.add_argument('--ate_task', type=int, default=1)
parser.add_argument('--qa_task', type=int, default=0)
parser.add_argument('--qa_model_path', type=str)
parser.add_argument("--do_predict", default=False, action='store_true')
parser.add_argument("--resume_training", default=False, action='store_true')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--n_epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=25)
parser.add_argument('--lr', type=int, default=3e-5)

args = parser.parse_args()

print()
print("="*25, f"Run experiment on *{args.dataset}* dataset", "="*25)
set_seed(args.random_seed)

# ---------------------------------------------------------------
# load dataset
print("\n>>> Loading data...")

data_paths = {'train': f'data/conll/{args.dataset}_train.txt',
              'test': f'data/conll/{args.dataset}_test.txt'}
data_bundle = ABSAQAPipe().process_from_file(data_paths)
print(data_bundle)

print("The target vocabs are:")
for tgt_vocab in ['ate_target', 'asc_target', 'unified_target']:
    print(tgt_vocab, ':', data_bundle.get_vocab(tgt_vocab))

# define vocab
unified_target_vocab = data_bundle.get_vocab('unified_target')
ate_target_vocab = data_bundle.get_vocab('ate_target')
asc_target_vocab = data_bundle.get_vocab('asc_target')
word_vocab = data_bundle.get_vocab("words")

# split the data
print("\n>>> Split the dev data...")
train_data = data_bundle.get_dataset('train')
train_data, dev_data = train_data.split(0.2)
test_data = data_bundle.get_dataset('test')

print("Number of QA pairs in each data split:")
print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")


# ---------------------------------------------------------------
# Load/initatize embeddings
print("\n>>> Load embedding...")

if args.embed == 'bert':
    bert_model_name = 'bert-base-chinese'
    bert_embed_dir = args.bert_dir + bert_model_name

    embed = BertEmbedding(vocab=data_bundle.get_vocab('words'),
                          model_dir_or_name=bert_embed_dir)
    print(f"Using `{bert_model_name}` as backbone (will take more time to train/eval)")

elif args.embed == 'fasttext':
    embed = StaticEmbedding(vocab=data_bundle.get_vocab('words'),
                            model_dir_or_name=args.fasttext_dir)
    print(f"Using pre-trained {args.embed}-300d word vectors")

else:
    raise Exception("The BERT dir needs to be provided!")


# ---------------------------------------------------------------
# define the model
print("\n>>> Set up model...")

ate_task = True if args.ate_task == 1 else False
qa_task = True if args.qa_task == 1 else False

model_type = f'model-{args.dataset}-ate-{args.ate_task}-qa-{args.qa_task}'
output_dir = f'outputs/{model_type}'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

print(f"Run Base model with ATE task being *{ate_task}* and QA pre-training being *{qa_task}*.\n")

model = Model(embed, embed_dropout=0.1,
              context_encoder=False, context_hidden_dim=100,
              inter_attn_heads=12, intra_attn_heads=12,
              a_enc_hidden_dim=300, a_enc_trans=True, a_enc_output_dim=64,
              kernel_size=3, d_self_hidden=256, d_local_hidden=256,
              ate_task=ate_task, append_a=True, tagger_dropout=0.1,
              ate_target_vocab=ate_target_vocab,
              asc_target_vocab=asc_target_vocab,
              unified_target_vocab=unified_target_vocab)

# Load pre-trained QA matching model
if qa_task and not args.resume_training:
    print("Load pre-trained QA matching model...")

    if args.qa_model_path:
        trained_qa_path = args.qa_model_path
    else:
        trained_qa_path = f'outputs/trained-qa-match-model-{args.dataset}-{args.embed}.pt'

    if os.path.exists(trained_qa_path):
        params = torch.load(trained_qa_path)
        # only initialize the QA attention module
        embed_keys = [k for k in list(params.keys()) if k.startswith('embedding')]
        for key in embed_keys:
            del params[key]
        model.load_state_dict(params, strict=False)
    else:
        raise Exception("You need to have the trained QA model first")

if args.resume_training:
    model.load_state_dict(torch.load(f'{output_dir}/trained_model.pt'))


# ---------------------------------------------------------------
print("\n>>> Begin Training...")

if args.qa_task == 0:
    n_epochs = 30  # already enough to converge
else:
    n_epochs = 40

if args.ate_task == 1 and args.qa_task == 0:
    lr = 4e-5
else:
    lr = 3e-5

optimizer = optim.Adam(model.parameters(), lr=lr)
metric = SpanFPreRecMetric(tag_vocab=unified_target_vocab, pred='pred',
                           target='unified_target', seq_len='q_len')
if args.gpu < 0:
    device = check_gpu()
else:
    device = args.gpu

clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = EvaluateCallback(test_data)
callbacks = [evaluate_callback, clip_callback]

warmup_steps = 0.1
if warmup_steps > 0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)

print(f"Configs: lr = {lr}, n_epochs = {n_epochs}, batch_size = {args.batch_size}")

trainer = Trainer(train_data, model,
                  optimizer=optimizer,
                  loss=LossInForward(),
                  sampler=BucketSampler(seq_len_field_name='q_len'),
                  batch_size=args.batch_size, n_epochs=n_epochs,
                  dev_data=dev_data, validate_every=-1,
                  metrics=metric, metric_key='f',
                  callbacks=callbacks,
                  device=device)

trainer.train(load_best_model=True)


# ---------------------------------------------------------------
print()
print('-'*30)
print(">>> Begin Testing...")
print(model_type)
tester = Tester(test_data, model, metrics=metric)
tester_results = tester.test()
print('-'*30)

# write results to file
with open(f"{output_dir}/results.txt", "a") as f:
    f.write(f'\n\n----- Results with random seed {args.random_seed} ------\n')
    f.write(str(tester_results))


# write predictions to file
if args.do_predict:
    print("\n>>> Write predictions...")
    predict_bs = 32
    predict_file_dir = f'{output_dir}/preds.txt'
    write_predictions(model, test_data, 'q_len', predict_bs, 
                      word_vocab, unified_target_vocab, predict_file_dir)

# save model
print("\n>>> Save trained model...")
trained_model_path = f'{output_dir}/trained_model.pt'
torch.save(model.state_dict(), trained_model_path)
print(f"Done! Saved to {trained_model_path}")
