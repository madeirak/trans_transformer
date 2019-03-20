import os
from tqdm import tqdm
from tools.langconv import *


with open('cmn.txt', 'r', encoding='utf8') as f:
    data = f.readlines()
    with  open("cn2en.txt", "w",encoding='utf-8') as ff:
        for line in tqdm(data[:]):
            [en, cn] = line.strip('\n').split('\t')
            cn = Converter('zh-hans').convert(cn)
            ff.writelines(cn+'\t'+en+'\n')
