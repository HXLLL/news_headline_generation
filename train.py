import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
import os

flag = tf.flags

flag.DEFINE_string("word2vec_norm", "embedding/word2vec_norm.txt", "pre-trained embeddings")
flag.DEFINE_string("data_path", "news_sample", "train data") # use sample when test
flag.DEFINE_string("content_file", "content.txt", "content file")
flag.DEFINE_string("title_file", "title.txt", "title file")
flag.DEFINE_string("file_encoding", "utf-8", "file encoding")
flag.DEFINE_integer("embedding_dim", 300, "embedding dim")
flag.DEFINE_integer("max_input_length", 20, "max input length")
flag.DEFINE_integer("max_output_length", 20, "max output length")

FLAGS = flag.FLAGS
print("Parameters:")
for k,v in FLAGS.flag_values_dict().items():
    print("{}={}".format(k,v))

IDeos = 1
IDsos = 2

class Config:
    batch_size = 192 # 384 in paper
    hidden_node = 300 # 600 in paper
    layer = 3 # 4 in paper
    learning_rate=0.01 # 0.01 in paper
    trainer = tf.train.RMSPropOptimizer(learning_rate = learning_rate, momentum=0.9, decay=0.9) # RMSProp in paper
    #trainer = tf.train.AdamOptimizer(learning_rate = learning_rate) # RMSProp in paper
    max_epoch = 1
    inferences = False
    output_vocab_size = 10000+2

class Data:
    def __init__(self, articles, len_articles, titles, len_titles):
        self.articles = np.array(articles)
        self.len_articles = np.array(len_articles)
        self.titles = np.array(titles)
        self.len_titles = np.array(len_titles)
    def shuffle(self):
        n = len(self.articles)
        id = list(range(n))
        random.shuffle(id)
        self.articles = self.articles[id]
        self.len_articles = self.len_articles[id]
        self.titles = self.titles[id]
        self.len_titles = self.len_titles[id]
    def length(self):
        return len(self.articles)
    def fit_shape(self):
        n = len(self.articles)
        max_len = np.max(self.len_titles)
        self.titles = self.titles[:,:max_len]

def make_vocab(filename):
    if filename:
        print("loading embedding file...")
        with open(filename, "r", encoding=FLAGS.file_encoding) as f:
            header = f.readline()
            vocab_size,layer_size = map(int, header.split(' '))

            dictionary = dict()
            dictionary['<eos>'] = IDeos
            dictionary['<sos>'] = IDsos
            vocab_size += 3
            init_W = np.array(np.random.uniform(-0.25,0.25,[vocab_size, FLAGS.embedding_dim]),dtype=np.float32)
            while(True):
                line = f.readline()
                if not line:
                    break
                word = line.split(' ')[0]
                dictionary[word] = len(dictionary)
                init_W[dictionary[word]] = np.array(line.split(' ')[1:], dtype=np.float32)
            
            return dictionary, init_W

def get_input(article_file, title_file):
    f = open(article_file, "r", encoding=FLAGS.file_encoding)
    g = open(title_file, "r", encoding=FLAGS.file_encoding)
    articles = []
    titles = []
    len_articles = []
    len_titles = []
    #set_trace()
    while True:
        article = f.readline().strip('\n')
        title = g.readline().strip('\n')
        title = title
        #title = '<sos> ' + title + ' <eos>'
        if not article: break
        if not title: break
        _ = [dictionary[word] for word in article.split(' ') if word in dictionary][:FLAGS.max_input_length]
        len_articles.append(len(_))
        _ += [0]*(FLAGS.max_input_length-len(_))
        articles.append(_)
    #    if (len(articles)==383):
    #        set_trace()
    #        print(_,article)
        
        _ = [dictionary[word] for word in title.split(' ') if word in dictionary][:FLAGS.max_output_length]
        len_titles.append(len(_))
        _ += [0]*(FLAGS.max_output_length-len(_))
        #_ += [IDeos]*(FLAGS.max_output_length-len(_))
        titles.append(_)

    return Data(articles, len_articles, titles, len_titles)

def load_batch(input, begin, end):
    articles = input.articles[begin:end]
    len_articles =  input.len_articles[begin:end]
    titles =  input.titles[begin:end]
    len_titles =  input.len_titles[begin:end]
    dat = Data(articles, len_articles, titles, len_titles)
    dat.fit_shape()
    return dat

dictionary, W = make_vocab(FLAGS.word2vec_norm)
id2word = {}
for i,v in dictionary.items():
    id2word[v] = i

config = Config()

input_articles = tf.placeholder(dtype=tf.int32, shape=[None,FLAGS.max_input_length])
articles_length = tf.placeholder(dtype=tf.int32, shape=[None])
input_titles = tf.placeholder(dtype=tf.int32, shape=[None, None])
titles_length = tf.placeholder(dtype=tf.int32, shape=[None])

encoder_embedding = tf.get_variable(name="encoder_embedding", dtype=tf.float32, initializer=W)
encoder_input = tf.nn.embedding_lookup(encoder_embedding, input_articles)
decoder_embedding = tf.get_variable(name="decoder_embedding", dtype=tf.float32, initializer=W)
decoder_input = tf.nn.embedding_lookup(decoder_embedding, input_titles)

def get_cell():
    return tf.nn.rnn_cell.BasicLSTMCell(config.hidden_node)
encoder_cell = tf.nn.rnn_cell.MultiRNNCell([get_cell() for i in range(config.layer)])
#encoder_cell = get_cell()

encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_input,sequence_length=articles_length,dtype=tf.float32)

projection_layer = tf.layers.Dense(config.output_vocab_size, use_bias=True, activation=tf.nn.sigmoid)

attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
    num_units=config.hidden_node,
    memory=encoder_output,
    memory_sequence_length=articles_length)
decoder_cell = tf.nn.rnn_cell.MultiRNNCell([get_cell() for i in range(config.layer)])
#decoder_cell = get_cell()
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    cell=decoder_cell,
    attention_mechanism=attention_mechanism,
    attention_layer_size=config.hidden_node)
train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_input, titles_length)
decoder_initial_state = decoder_cell.zero_state(batch_size=config.batch_size,dtype=tf.float32).clone(cell_state=encoder_state)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper, decoder_initial_state, output_layer=projection_layer)
decoder_output, decoder_state, decoder_output_length = tf.contrib.seq2seq.dynamic_decode(decoder)
#set_trace()

logits = decoder_output.rnn_output

#cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_output.)
#set_trace()
#cur_len = tf.to_int32(logits.shape[1])
title_onehot = tf.one_hot(input_titles, config.output_vocab_size, dtype=tf.float32)
title_mask = tf.to_float(tf.sequence_mask(titles_length))
loss = tf.contrib.seq2seq.sequence_loss(logits, input_titles, title_mask)
#title_onehot = tf.slice(title_onehot, begin=[0,0], size=[-1, cur_len])
#loss = -tf.reduce_mean(tf.reduce_sum(tf.log(tf.reduce_sum(title_onehot*logits, axis=-1)),axis=-1))

print("loading data for training...")
train_op = config.trainer.minimize(loss)
train_content_path = os.path.join(FLAGS.data_path, FLAGS.content_file)
train_title_path = os.path.join(FLAGS.data_path, FLAGS.title_file)
train_data = get_input(train_content_path, train_title_path)

set_trace()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.667)
se = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

print("trainable variables:")
print(tf.trainable_variables())
se.run(tf.global_variables_initializer())

# =============== train =================
#def train():
print("\ntraining...")
#set_trace()
for epoch in range(config.max_epoch):
    train_data.shuffle()
    num_batch = train_data.length() // config.batch_size
    #num_batch = 1
    print("epoch %d: %d batches in total" % (epoch, num_batch))
    epoch_loss = []
    for i in range(num_batch):
        batch_data = load_batch(train_data, i*config.batch_size, (i+1)*config.batch_size)
        batch_loss, _ = se.run([loss, train_op], feed_dict = {
            input_articles: batch_data.articles,
            articles_length: batch_data.len_articles,
            input_titles: batch_data.titles,
            titles_length: batch_data.len_titles
        })
        pred = se.run([decoder_output.sample_id], feed_dict = {
            input_articles: batch_data.articles,
            articles_length: batch_data.len_articles,
            input_titles: batch_data.titles,
            titles_length: batch_data.len_titles
        })
        print(pred[0])
        epoch_loss.append(batch_loss)
        print("epoch %d batch %d: loss %lf" % (epoch, i, batch_loss))

    print("epoch %d: loss %lf" % (epoch, np.mean(epoch_loss)))

#train()

# =============== test ===================
print("\ntesting...")

test_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, tf.fill([1], IDsos), IDeos)
test_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, test_helper, encoder_state, output_layer=projection_layer)
outputs, _, output_len = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=FLAGS.max_output_length)
out_title = outputs.sample_id
test_data = get_input('news_sample/content_test.txt', 'news_sample/title.txt')
set_trace()
num_batch = test_data.length() // config.batch_size
for i in range(num_batch):
    batch_data = load_batch(test_data, i*config.batch_size, (i+1)*config.batch_size)
    ress = se.run(out_title, feed_dict={
        input_articles: batch_data.articles,
        articles_length: batch_data.len_articles,
        input_titles: batch_data.titles,
        titles_length: batch_data.len_titles
    })
    for res in ress:
        print(res, end=' : ')
        for word in res:
            print("%s" % id2word[word], end=' ')
        print()