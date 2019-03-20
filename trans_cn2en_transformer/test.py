from utils import create_hparams,make_vocab,en_segment,cn_segment
from transformer import Graph
import tensorflow as tf
from tqdm import tqdm
import numpy as np
#from train import data_count

'''需与train.py中data_count相同'''
data_count = 20000

#make_vocab
with open('cn2en.txt', 'r', encoding='utf8') as f:
    data = f.readlines()
    inputs = []
    outputs = []
    for line in tqdm(data[:data_count]):
        [en, cn] = line.strip('\n').split('\t')

        inputs.append(en.replace(',', ' ,')[:-1].lower())  # 句中逗号后本有空格，在逗号前增加空格，然后将逗号按一个元素分隔，去掉句末标点，转为小写
        outputs.append(cn[:-1])  # 去掉汉语标签句末标点
    # print(inputs[:10])
    # print(outputs[274:276])
    outputs = en_segment(outputs)
    inputs = cn_segment(inputs)
    # print(outputs)

encoder_vocab,decoder_vocab = make_vocab(inputs,outputs)
print('\n-----------vocab have made-----------')




arg = create_hparams()
arg.is_training = False
arg.input_vocab_size = len(encoder_vocab)
arg.label_vocab_size = len(decoder_vocab)


g = Graph(arg)

saver =tf.train.Saver()
with tf.Session() as sess:
    latest = tf.train.latest_checkpoint('model')  # 查找最新保存的检查点文件的文件名，latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest)  # restore(sess,save_path)，需要启动图表的会话。
    # 该save_path参数通常是先前从save()调用或调用返回的值latest_checkpoint()

    print('输入exit，敲回车结束.')
    while True:
        line = input('输入测试汉语: ')
        if line == 'exit': break
        #print(line[-1])
        #if line[-1] == ',' or '.' or '?' or '!'or'。'or'，'or'！':
        #    line = line[:-1]
        #print(line)
        line = line.replace('，', ' ,').strip('\n').split(' ')
        x = np.array([encoder_vocab.index(hanzi) for hanzi in line])
        x = x.reshape(1, -1)
        de_inp = [[decoder_vocab.index('<GO>')]]  #de_inp  ->  decoder_inputs
        while True:
            y = np.array(de_inp)
            preds = sess.run(g.preds, {g.x: x, g.de_inp: y})
            #print(preds)
            if preds[0][-1] == decoder_vocab.index('<EOS>'):
                break
            de_inp[0].append(preds[0][-1])
        got = ' '.join(decoder_vocab[idx] for idx in de_inp[0][1:])
        print(got)