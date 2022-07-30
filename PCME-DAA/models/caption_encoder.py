""" Caption encoder based on PVSE implementation.
Reference code:
    https://github.com/yalesong/pvse/blob/master/model.py
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchtext

from models.pie_model import PIENet
from models.uncertainty_module import UncertaintyModuleText
from utils.tensor_utils import l2_normalize, sample_gaussian_tensors

from transformers import BertModel


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0).to(lengths.device)
    mask = (ind >= lengths.unsqueeze(1)) if set_pad_to_one \
        else (ind < lengths.unsqueeze(1))
    mask = mask.to(lengths.device)
    return mask
    

class EncoderText(nn.Module):
    def __init__(self, opt):
        super(EncoderText, self).__init__()

        wemb_type, word_dim, embed_dim = \
            opt.wemb_type, opt.word_dim, opt.embed_dim

        self.embed_dim = embed_dim
        self.use_attention = opt.txt_attention
        self.use_probemb = opt.get('txt_probemb')

        # Word embedding
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(word_dim, embed_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, embed_dim))
        self.relu = nn.ReLU()

        self.bert_freeze(self.bert, True)

        if self.use_attention:
            self.pie_net = PIENet(1, word_dim, embed_dim, word_dim // 2)

        self.uncertain_net = UncertaintyModuleText(word_dim, embed_dim, word_dim // 2)
        self.init_weights()

        self.n_samples_inference = opt.get('n_samples_inference', 0)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def bert_freeze(self, bert_model, txt_finetune=False):
        unfreeze_layers = ['bert.pooler','out.']
        for name ,param in bert_model.named_parameters():
            param.requires_grad = txt_finetune
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

    def forward(self, x, lengths):

        # Embed word ids to vectors
        bert_attention_mask = (x != 0)
        wemb_out = self.bert(x, bert_attention_mask)[0]  # B x N x D
        out = l2_normalize(self.fc(wemb_out))
        out = self.relu(out)

        out = self.avgpool(out).squeeze(1)

        bert_attention_mask = ~bert_attention_mask

        output = {}

        if self.use_attention:
            out, attn, residual = self.pie_net(out, wemb_out, bert_attention_mask)
            output['attention'] = attn
            output['residual'] = residual

        if self.use_probemb:
            if not self.use_attention:
                bert_attention_mask=None
            uncertain_out = self.uncertain_net(wemb_out, bert_attention_mask, lengths)
            logsigma = uncertain_out['logsigma']
            output['logsigma'] = logsigma
            output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)

        if self.use_probemb and self.n_samples_inference:
            output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_samples_inference)
        else:
            output['embedding'] = out

        return output
