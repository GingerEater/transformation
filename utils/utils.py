# -*- coding:utf-8 -*-

import json
import csv
# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
from xml.etree.ElementTree import parse
import xml.etree.cElementTree as ET
import glob
import codecs
from nltk.tokenize import sent_tokenize
import constant
import spacy

def prepare_data_for_stanford_parse():
    dirs = ['Train', 'Test_Gold']
    for dir in dirs:
        with open('../data_trans/standard_data/real_Restaurants_'+dir+'.tsv', 'r') as f:
            lines = f.readlines()
            tmp = []
            for line in lines:
                line = str(line).split('\t')[0]
                tmp.append(line+'\n')
        with open('../data_trans/raw/real_Restaurants_'+dir+'_raw.txt', 'w') as f:
            for line in tmp:
                f.write(line)


def process_stanford_data():
    dirs = ['real_Restaurants_Train_parse', 'real_Restaurants_Test_Gold_parse']
    for dir in dirs:
        count = 0
        with open('../data_trans/parse/'+dir+'.txt', 'r') as f:
            lines = f.readlines()
            flag = 'tag'
            tags = []; parsers = []
            the_line = ''; tmp = []
            for line in lines:
                if line == '\n' or line == '':
                    count += 1
                    if flag == 'tag':
                        tags.append(the_line)
                        flag = 'parser'
                        the_line = ''
                    elif flag == 'parser':
                        parsers.append(tmp)
                        tmp = []
                        flag = 'tag'
                else:
                    if flag == 'tag':
                        the_line = line.replace('\n', '')
                    elif flag == 'parser':
                        tmp.append(line.replace('\n', ''))

        res = []
        for tag, parser in zip(tags, parsers):
            tag = tag.split(' ')
            tokens = []; stanford_deprel = []; stanford_head = []; stanford_cur = []
            for item in tag:
                tokens.append(item.split('/')[0])
            
            for item in parser:
                tmp = item.split('(')

                relation = tmp[0]
                head = tmp[1].split(',')[0].split('-')[-1].replace("'", "")
                cur = tmp[1].split(',')[1].split('-')[-1].replace(')', '').replace("'", "")

                stanford_deprel.append(relation)
                stanford_head.append(head)
                stanford_cur.append(cur)

                assert len(stanford_deprel) == len(stanford_head)
                assert len(stanford_deprel) == len(stanford_cur)
            
            dic = {'tokens':tokens, 'stanford_deprel':stanford_deprel, 'stanford_head':stanford_head, 'stanford_cur':stanford_cur}
            res.append(dic)

        with open('../data_trans/json/'+dir+'.json', 'w') as f:
            json.dump(res, f)

def get_full_data():
    # dirs = ['real_Restaurants_Train_parse', 'real_Restaurants_Test_Gold_parse']
    dirs = ['real_Restaurants_Test_Gold_parse']
    for dir in dirs:
        print(dir)
        with open('../data_trans/json/'+dir+'.json', 'r') as f:
            data = json.load(f)

        gcn_lines = []
        for dic in data:
            tokens = dic['tokens']
            deprel = dic['stanford_deprel']
            head = dic['stanford_head']
            cur = dic['stanford_cur']
            
            gcn_lines.append( ( tokens, deprel, head,  cur ) )
        
        # 加上label
        raw_lines = []
        with open('../data_trans/standard_data/'+dir[:-6]+'.tsv', 'r') as f:
            data = f.readlines()
            for line in data:
                tmp = str(line).replace('\n','').split('\t')
                raw_lines.append((tmp[0].split(' '), tmp[1].split(' '), int(tmp[-1])))

        # 根据gcn_lines和raw_lines构造新的数据格式
        new_lines = []

        assert len(gcn_lines) == len(raw_lines)

        for gcn, raw in zip(gcn_lines, raw_lines):
            if len(raw[0]) != len(gcn[0]):
                print(' '.join(gcn[0]))
                print(' '.join(raw[0]))
                print(gcn[0])
                print(raw[0])
            assert len(raw[0]) == len(gcn[0])
            context = raw[0]; target = raw[1]; tgt_len = len(target)
            b_e = ''
            for j in range(len(context)):
                if context[j] == target[0] and context[j:j+tgt_len] == target:
                    b_e = str(j) + ' ' + str(j+tgt_len)
                    break
            if b_e == '':
                print(' '.join(context))
                print(' '.join(target))
            assert b_e != ''
            tmp = (' '.join(raw[0]), ' '.join(raw[1]), b_e, ' '.join(gcn[1]), ' '.join(gcn[2]), ' '.join(gcn[3]), raw[-1])
            new_lines.append(tmp)
        
        with open('../data_trans/res/'+dir[:-6]+'.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(new_lines)

def get_precision_test_data():
    dirs = ['real_Restaurants_All_new', 'real_Restaurants_Train_new', 'real_Restaurants_Test_Gold_new']
    for dir in dirs:
        with open('../data/'+dir+'.tsv', 'r') as f:
            lines = f.readlines()
            new_lines = []
            for line in lines:
                tmp = str(line).replace('\n', '').split('\t')
                new_lines.append((tmp[0], tmp[1], tmp[-1]))
        with open ('../data/'+dir+'_tmp.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(new_lines)


def add_tgt_be():
    dirs = ['real_Restaurants_Test_Gold_new.tsv']
    for dir in dirs:
        print(dir)
        with open('../data/'+dir, 'r') as f:
            lines = f.readlines()
            new_lines = []
            for i in range(len(lines)):
                tmp = lines[i].replace('\n', '').split('\t')
                context = tmp[0]; target = tmp[1]
                context = context.replace('-LRB-', '(').replace('-RRB-', ')')
                target = target.replace('-LRB-', '(').replace('-RRB-', ')').replace('(', ' ( ').replace(')', ' ) ').split(' ')
                target = [x for x in target if x != '']
                target = ' '.join(target)
                if target not in context:
                    print(lines[i])
                assert target in context
                context = context.split(' '); target = target.split(' '); tgt_len = len(target)
                b_e = ''
                for j in range(len(context)):
                    if context[j] == target[0] and context[j:j+tgt_len] == target:
                        b_e = str(j) + ' ' + str(j+tgt_len)
                        break
                if b_e == '':
                    print(lines[i])
                assert b_e != ''
                new_lines.append((tmp[0], tmp[1],  b_e, tmp[2], tmp[3], tmp[4], int(tmp[5])))

        with open('../data/'+dir, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(new_lines)


def check_gcn_data():
    with open('../data/real_Restaurants_Train_new.tsv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split('\t')
            tgt = tmp[1].split(' ')
            tgt_be = tmp[2].split(' ')
            if len(tgt) != int(tgt_be[1]) - int(tgt_be[0]):
                print (line)


def get_distance():
    dirs = ['real_Restaurants_All.tsv', 'real_Restaurants_Train.tsv', 'real_Restaurants_Test_Gold.tsv',
            'real_Laptops_All.tsv', 'real_Laptops_Train.tsv', 'real_Laptops_Test_Gold.tsv']
    for dir in dirs:
        print(dir)
        with open('../data/'+dir, 'r') as f:
            lines = f.readlines()
            new_lines = []
            for line in lines:
                tmp = line.replace('\n', '').split('\t')
                src = tmp[0].split(' '); b_e = tmp[2].split(' '); b_e = list(map(int, b_e))
                m = b_e[1] - b_e[0]; k = b_e[0]
                dis = [i for i in range(len(src)) ]
                for i in range(len(dis)):
                    if i < k+m:
                        dis[i] = k+m-i
                    elif i >= k+m:
                        dis[i] = i - k
                    else:
                        dis[i] = 0
                dis = list(map(str, dis))
                line = (tmp[0], tmp[1], tmp[2], ' '.join(dis),  int(tmp[-1]))
                new_lines.append(line)
        with open('../data/'+dir, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(new_lines)

if __name__ == '__main__':
    # print('hhhhhh')
    # prepare_data_for_stanford_parse()
    # process_stanford_data()
    # get_full_data()
    # check_error()
    # get_precision_test_data()
    # add_biaodian()
    # add_tgt_be()
    # check_gcn_data()
    get_distance()
