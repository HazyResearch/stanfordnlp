import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import stanfordnlp.models.depparse.mapping_utils as util
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

from stanfordnlp.models.common.biaffine import DeepBiaffineScorer
from stanfordnlp.models.common.hlstm import HighwayLSTM
from stanfordnlp.models.common.dropout import WordDropout
from stanfordnlp.models.common.vocab import CompositeVocab
from stanfordnlp.models.common.char_model import CharacterModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Parser(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, share_hid=False):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            # frequent word embeddings
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            self.lemma_emb = nn.Embedding(len(vocab['lemma']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim'] * 2

        if self.args['tag_emb_dim'] > 0:
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)

            if not isinstance(vocab['xpos'], CompositeVocab):
                self.xpos_emb = nn.Embedding(len(vocab['xpos']), self.args['tag_emb_dim'], padding_idx=0)
            else:
                self.xpos_emb = nn.ModuleList()

                for l in vocab['xpos'].lens():
                    self.xpos_emb.append(nn.Embedding(l, self.args['tag_emb_dim'], padding_idx=0))

            self.ufeats_emb = nn.ModuleList()

            for l in vocab['feats'].lens():
                self.ufeats_emb.append(nn.Embedding(l, self.args['tag_emb_dim'], padding_idx=0))

            input_size += self.args['tag_emb_dim'] * 2

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        if self.args['pretrain']:
            # pretrained embeddings, by default this won't be saved into model file
            add_unsaved_module('pretrained_emb', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        # recurrent layers
        self.parserlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True, dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.parserlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.parserlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        
        self.output_size = 400
        # classifiers

        self.hypmapping =  nn.Sequential(
                           nn.Linear(2*self.args['hidden_dim'], 1000).to(device),
                           nn.ReLU().to(device),
                           nn.Linear(1000, 100).to(device),
                           nn.ReLU().to(device),
                           nn.Linear(100, self.output_size).to(device),
                           nn.ReLU().to(device))

        # self.scale = nn.Parameter(torch.cuda.FloatTensor([1.0]), requires_grad=True)                  
        # self.unlabeled = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=args['dropout'])
        # self.deprel = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], len(vocab['deprel']), pairwise=True, dropout=args['dropout'])
        # if args['linearization']:
        #     self.linearization = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=args['dropout'])
        # if args['distance']:
        #     self.distance = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=args['dropout'])

        # criterion
        # self.crit = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum') # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, scale, root, subsample=True):
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        if self.args['pretrain']:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]


        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            lemma_emb = self.lemma_emb(lemma)
            lemma_emb = pack(lemma_emb)
            inputs += [word_emb, lemma_emb]

        if self.args['tag_emb_dim'] > 0:
            pos_emb = self.upos_emb(upos)

            if isinstance(self.vocab['xpos'], CompositeVocab):
                for i in range(len(self.vocab['xpos'])):
                    pos_emb += self.xpos_emb[i](xpos[:, :, i])
            else:
                pos_emb += self.xpos_emb(xpos)
            pos_emb = pack(pos_emb)

            feats_emb = 0
            for i in range(len(self.vocab['feats'])):
                feats_emb += self.ufeats_emb[i](ufeats[:, :, i])
            feats_emb = pack(feats_emb)

            inputs += [pos_emb, feats_emb]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
            inputs += [char_reps]


        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        # print("inputs", inputs)

        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)

        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)
        # print("batch size", inputs[0].batch_sizes)
        # print("lstm inputs", lstm_inputs)

        # print("word size", word.size(0))
        lstm_outputs, _ = self.parserlstm(lstm_inputs, sentlens, hx=(self.parserlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), self.parserlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
        lstm_outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True)
        lstm_outputs_normalized = torch.zeros(lstm_outputs.shape, device=device)
        # print("lstm shape", lstm_outputs.shape)

        #This can be done without a for loop.
        for idx in range(lstm_outputs.shape[0]):
            embedding = lstm_outputs[idx]
            norm = embedding.norm(p=2, dim=1, keepdim=True)
            max_norm = torch.max(norm)+1e-3
            normalized_emb = embedding.div(max_norm.expand_as(embedding))
            lstm_outputs_normalized[idx] = normalized_emb

        # print("After normalization:", lstm_outputs.shape)
        
        lstm_postdrop = self.drop(lstm_outputs_normalized)
        mapped_vectors = self.hypmapping(lstm_postdrop)
        subsample_ratio = 1.0
        preds = []
        # print("subsample ratio", subsample_ratio)
        edge_acc = 0.0
        # f1_total, correct_heads, node_system, node_gold = 0, 0, 0, 0

        if self.training:
            unlabeled_target = head
            n = unlabeled_target.shape[1]
            sampled_rows = list(range(n))
            dist_recovered = util.distance_matrix_hyperbolic_batch(mapped_vectors, sampled_rows, scale)
            # print("dist recovered shape", dist_recovered.shape)            
            dummy = dist_recovered.clone()
            target_dummy = unlabeled_target.clone()
            edge_acc = util.compare_mst_batch(target_dummy.cpu().numpy(), dummy.detach().cpu().numpy())
            
            loss = util.distortion_batch(unlabeled_target.contiguous(), dist_recovered, n, sampled_rows)

        else:
            loss = 0
            unlabeled_target = head
            # print("target shape", unlabeled_target.shape)
            n = unlabeled_target.shape[1]
            sampled_rows = list(range(n))
            
            # print("sampled rows", sampled_rows)
            # print("mapped vectors", mapped_vectors.shape)
            dist_recovered = util.distance_matrix_hyperbolic_batch(mapped_vectors, sampled_rows, scale)
            # print("dist recovered shape", dist_recovered.shape)            
            dummy = dist_recovered.clone()
            target_dummy = unlabeled_target.clone()
            preds = util.get_heads_batch(dummy.detach().cpu().numpy(), sentlens, root)
            # preds.append(F.log_softmax(unlabeled_scores, 2).detach().cpu().numpy())
            # preds.append(deprel_scores.max(3)[1].detach().cpu().numpy())

        return loss, edge_acc, preds
