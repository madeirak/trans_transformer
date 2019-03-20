import tensorflow as tf
from tqdm import tqdm
from utils import en_segment,cn_segment,get_vocab,get_batch,make_vocab,data_format,create_hparams
import os
from transformer import Graph


epoch = 10#20
batch_size = 32#64
data_count = 20000     #取训练集中的数据句数


with open('cn2en.txt', 'r', encoding='utf8') as f:
    data = f.readlines()
    inputs = []
    outputs = []
    for line in tqdm(data[:data_count]):
        [en, cn] = line.strip('\n').split('\t')

        outputs.append(cn[:-1])  # 去掉汉语标签句末标点
        inputs.append(en.replace(',', ' ,')[:-1].lower())  # 句中逗号后本有空格，在逗号前增加空格，然后将逗号按一个元素分隔，去掉句末标点，转为小写

    # print(inputs[:10])
    # print(outputs[274:276])
    inputs = cn_segment(inputs)
    outputs = en_segment(outputs)
    # print(outputs)


encoder_vocab,decoder_vocab = make_vocab(inputs,outputs)
print('\n-----------vocab have made-----------')

encoder_inputs, decoder_inputs, decoder_targets = data_format(inputs,outputs,encoder_vocab,decoder_vocab)



arg = create_hparams()
arg.input_vocab_size = len(encoder_vocab)
arg.label_vocab_size = len(decoder_vocab)
arg.epochs = epoch
arg.batch_size = batch_size

g = Graph(arg)

saver =tf.train.Saver()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    add_num = 0
    if os.path.exists('model/checkpoint'):
        print('loading  model...')
        latest = tf.train.latest_checkpoint('model')  # 查找最新保存的检查点文件的文件名，latest_checkpoint(checkpoint_dir)
        add_num = int(latest.split('_')[-1])
        saver.restore(sess, latest)#该save_path参数通常是先前从save()调用或调用返回的值 latest_checkpoint()
    writer = tf.summary.FileWriter('model/tensorboard', tf.get_default_graph())
    for k in range(epoch):
        if k == 0 : print('\n-epoch',k +add_num+ 1, ':')
        else :      print('\n-epochs', k +add_num+ 1, ':')
        total_loss = 0
        batch_num = len(encoder_inputs) // arg.batch_size
        batch = get_batch(encoder_inputs, decoder_inputs, decoder_targets, arg.batch_size)
        for i in tqdm(range(batch_num)):
            #print('--------This is No.'+str(i+1)+'batch of No.'+str(k+1)+'epoch.--------')
            encoder_input, decoder_input, decoder_target = next(batch)
            feed = {g.x: encoder_input, g.y: decoder_target, g.de_inp:decoder_input}
            cost,_ = sess.run([g.mean_loss,g.train_op], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs=sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)

        if (k+1) % 5 == 0:
            print('epochs', k+1, ': average loss = ', total_loss/batch_num)
    saver.save(sess, 'model/model_%d' % (epoch + add_num))
    writer.close()

