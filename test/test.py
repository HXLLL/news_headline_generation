import tensorflow as tf
import BLEU

test_size=88 #测试集大小
IDeos=0 #结束标志
IDsos=1 #开始标志
max_decode_len=20 #最大解码长度
maxlen=200 #文本最大长度
vocab_size=3000 #词库大小
emb_dim=50 #词向量维度
emb_shape=[vocab_size,emb_dim] #词向量矩阵的形状
num_units=150 #LSTM节点数
inference=True

X = tf.placeholder(tf.int32,[None,maxlen]) #test_size个文本
LEN = tf.placeholder(tf.int32,[None]) #每个文本的长度

embedding = tf.get_variable("my_emb",initializer=tf.truncated_normal(shape=emb_shape),trainable=True)
#词向量矩阵，随机初始化，可训练
input = tf.nn.embedding_lookup(embedding,X)

encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    cell=encoder_cell,
    inputs=input,
    sequence_length=LEN,
    time_major=False,
    dtype=tf.float32)
#sequence_length：对每个batch只编码到对应句长
#time_major：为False时output具有[batch_size,max_time,num_units]的形状，Attention的memory需要这种形状
#greedy似乎只支持time_major=False

attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
    num_units=num_units,
    memory=encoder_outputs,
    memory_sequence_length=LEN)
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    cell=cell,
    attention_mechanism=attention_mechanism,
    attention_layer_size=num_units)
decoder_initial_state = decoder_cell.zero_state(batch_size=test_size,dtype=tf.float32).clone(cell_state=encoder_state)
projection_layer = tf.layers.Dense(units=vocab_size,use_bias=False)
#decode加入attention机制
#全连接层num_units -> vocab_size，将cell的输出态转化为每个词的选择概率

helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    embedding=embedding,
    start_tokens=tf.fill([test_size],IDsos),
    end_token=IDeos)
#embedding_matrix：词向量矩阵，每次解码出ID的embedding作为下一个输入
#start_tokens：batch_size*1，标志每个batch开始解码时传入的ID
#end_token：结束标志，当解码出结束标志时停止解码

decoder = tf.contrib.seq2seq.BasicDecoder(
    cell=decoder_cell,
    helper=helper,
    initial_state=decoder_initial_state,
    output_layer=projection_layer)

outputs = tf.contrib.seq2seq.dynamic_decode(
    decoder=decoder,
    maximum_iterations=max_decode_len)

print(outputs)
pred = outputs[0].sample_id
print(pred)
#outputs[0]分为两部分：
#rnn_output：每一步每个词的选择概率
#sample_id：每一步真正选择的词的ID

"""
sess = tf.Session( )
pred = sess.run(fetches=pred,feed_dict={X:test_X,LEN:test_len})
print(BLEU.calc(pred,truth,IDeos))
"""