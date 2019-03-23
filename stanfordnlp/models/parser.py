"""
Entry point for training and evaluating a dependency parser.

This implementation combines a deep biaffine graph-based parser with linearization and distance features.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

"""
Training and evaluation for the parser.
"""

import sys
import os
import shutil
import time
from datetime import datetime
import argparse
import logging
import numpy as np
import random
import torch
from torch import nn, optim

from stanfordnlp.models.depparse.data import DataLoader
from stanfordnlp.models.depparse.trainer import Trainer
from stanfordnlp.models.depparse import scorer
from stanfordnlp.models.common import utils
from stanfordnlp.models.common.pretrain import Pretrain

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/depparse', help='Root dir for saving models.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/word2vec', help='Directory of word vectors')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--gold_file', type=str, default=None, help='Output CoNLL-U file.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--char_hidden_dim', type=int, default=400)
    # parser.add_argument('--deep_biaff_hidden_dim', type=int, default=200)
    # parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=200)
    parser.add_argument('--word_emb_dim', type=int, default=75)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--output_size', type=int, default=400)
    parser.add_argument('--tag_emb_dim', type=int, default=50)
    parser.add_argument('--transformed_dim', type=int, default=125)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--word_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--subsample_ratio', type=float, default=1.0)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    parser.add_argument('--no_linearization', dest='linearization', action='store_true', help="Turn off linearization term.")
    parser.add_argument('--no_distance', dest='distance', action='store_true', help="Turn off distance term.")
    parser.add_argument('--sample_dev', type=float, default=0.1, help='Subsample dev data.')
    parser.add_argument('--sample_train', type=float, default=0.01, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='sgd', help='sgd, rsgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--max_steps_before_stop', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=2, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models/depparse', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)

    formatter = logging.Formatter('%(asctime)s %(message)s')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        datefmt='%FT%T',)
    logging.info(f"Logging")
    log = logging.getLogger()
    fh  = logging.FileHandler("logparsing")
    fh.setFormatter(formatter)
    log.addHandler(fh)

    args = vars(args)
    print("Running parser in {} mode".format(args['mode']))

    # if args['mode'] == 'train':
    train(args)
    # else:
    #     evaluate(args)

def train(args):
    utils.ensure_dir(args['save_dir'])
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
            else '{}/{}_parser.pt'.format(args['save_dir'], args['shorthand'])

    # load pretrained vectors
    vec_file = utils.get_wordvec_file(args['wordvec_dir'], args['shorthand'])
    pretrain_file = '{}/{}.pretrain.pt'.format(args['save_dir'], args['shorthand'])
    pretrain = Pretrain(pretrain_file, vec_file)

    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    train_batch = DataLoader(args['train_file'], args['batch_size'], args, pretrain, evaluation=False)
    vocab = train_batch.vocab
    # dev_batch = DataLoader(args['eval_file'], args['batch_size'], args, pretrain, vocab=vocab, evaluation=True)

    # pred and gold path
    system_pred_file = args['output_file']
    gold_file = args['gold_file']

    # skip training if the language does not have training or dev data
    # if len(train_batch) == 0 or len(dev_batch) == 0:
    if len(train_batch) == 0:
        print("Skip training because no data available...")
        sys.exit(0)

    current_lr = args['lr']
    scale_lr = current_lr
    print("Training parser...")
    trainer = Trainer(args=args, vocab=vocab, pretrain=pretrain, use_cuda=args['cuda'])
    print("optimizer:", trainer.optimizer)
    print("mapping optimizer:", trainer.mapping_optimizer)
    print("scale optimizer:", trainer.scale_optimizer)
    global_step = 0
    max_steps = args['max_steps']
    dev_score_history = []
    best_dev_preds = []
    global_start_time = time.time()
    format_str = '{}: step {}/{}, loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

    last_best_step = 0
    # start training
    train_loss = 0

    while True:
        do_break = False
        for i, batch in enumerate(train_batch):
            for iter in range(2000):
                start_time = time.time()
                global_step += 1
                loss, _ = trainer.update(batch, eval=False, subsample=True) # update step
                train_loss += loss
                duration = time.time() - start_time
                avg_loss = loss/args['log_step']
                logging.info(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step,\
                        max_steps, avg_loss, duration, current_lr))
                    

                if global_step % args['eval_interval'] == 0:
                #     # eval on dev
                #     print("Evaluating on dev set...")
                    # dev_acc_total = 0
                    # for batch in dev_batch:
                    #     dev_acc = trainer.predict(batch)
                    #     dev_acc_total += dev_acc


                #     dev_batch.conll.set(['head', 'deprel'], [y for x in dev_preds for y in x])
                #     dev_batch.conll.write_conll(system_pred_file)
                #     _, _, dev_score = scorer.score(system_pred_file, gold_file)
                    full_loss, edge_acc = trainer.update(batch, eval=False, subsample=False)
                    # dev_acc_total /= len(dev_batch)
                    # print("step {}: Full loss = {:.6f}, Edge acc. = {:.4f}".format(global_step, full_loss, edge_acc))
                    logging.info("step {}: Full loss = {:.6f}, Edge acc. = {:.4f}".format(global_step, full_loss, edge_acc))
                    # print("Dev accuracy", dev_acc_total)
                    # logging.info("step {}: Dev acc. = {:.6f}".format(global_step, dev_acc_total))
                # train_loss = 0

                if (global_step % 1000 == 0) and current_lr>1e-5:
                    current_lr *= 0.5
                    scale_lr *= 0.5
                    trainer.optimizer = utils.RiemannianSGD(trainer.model.parameters(), lr=current_lr, rgrad=utils.poincare_grad, retraction=utils.retraction)
                    trainer.scale_optimizer = torch.optim.SGD([trainer.scale], lr=scale_lr)
                #     # save best model
                #     if len(dev_score_history) == 0 or dev_score > max(dev_score_history):
                #         last_best_step = global_step
                #         trainer.save(model_file)
                #         print("new best model saved.")
                #         best_dev_preds = dev_preds

                #     dev_score_history += [dev_score]
                #     print("")

                if global_step - last_best_step >= args['max_steps_before_stop']:
                    # if not using_amsgrad:
                    #     print("Switching to AMSGrad")
                    #     last_best_step = global_step
                    #     using_amsgrad = True
                    #     trainer.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args['lr'], betas=(.9, args['beta2']), eps=1e-6)
                    # else:
                    do_break = True
                    break

                if global_step >= args['max_steps']:
                    do_break = True
                    break

        if do_break: break
        # print("Reshuffling now")
        # train_batch.reshuffle()

    print("Training ended with {} steps.".format(global_step))

    # best_f, best_eval = max(dev_score_history)*100, np.argmax(dev_score_history)+1
    # print("Best dev F1 = {:.2f}, at iteration = {}".format(best_f, best_eval * args['eval_interval']))

def evaluate(args):
    # file paths
    system_pred_file = args['output_file']
    gold_file = args['gold_file']
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
            else '{}/{}_parser.pt'.format(args['save_dir'], args['shorthand'])
    pretrain_file = '{}/{}.pretrain.pt'.format(args['save_dir'], args['shorthand'])
    
    # load pretrain
    pretrain = Pretrain(pretrain_file)

    # load model
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(pretrain=pretrain, model_file=model_file, use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab

    # load config
    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand'] or k == 'mode':
            loaded_args[k] = args[k]

    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    batch = DataLoader(args['eval_file'], args['batch_size'], loaded_args, pretrain, vocab=vocab, evaluation=True)

    if len(batch) > 0:
        print("Start evaluation...")
        preds = []
        for i, b in enumerate(batch):
            preds += trainer.predict(b)
    else:
        # skip eval if dev data does not exist
        preds = []

    # write to file and score
    batch.conll.set(['head', 'deprel'], [y for x in preds for y in x])
    batch.conll.write_conll(system_pred_file)

    if gold_file is not None:
        _, _, score = scorer.score(system_pred_file, gold_file)

        print("Parser score:")
        print("{} {:.2f}".format(args['shorthand'], score*100))

if __name__ == '__main__':
    main()
