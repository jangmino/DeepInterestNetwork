import tensorflow as tf

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell

class Model(object):

  def __init__(self, arch_dict, D = 100, slot_dim = 100):

    self.i = tf.placeholder(tf.float32, [None, D]) # [B, D]
    self.j = tf.placeholder(tf.float32, [None, D]) # [B, D]
    self.y = tf.placeholder(tf.float32, [None,]) # [B]
    self.hist_i = tf.placeholder(tf.float32, [None, D]) # [B, D]
    self.lr = tf.placeholder(tf.float64, [])
    self.phase = tf.placeholder(tf.bool, name="pholder_phase")

    hidden_units = arch_dict['emb_dim']
    f1_dim = arch_dict['f1_dim']
    f2_dim = arch_dict['f2_dim']

    B = tf.shape(self.u)[0]

    if use_i_emb:
      item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
      self.item_b = tf.get_variable("item_b", [item_count],
                             initializer=tf.constant_initializer(0.0))

    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2 if use_i_emb else hidden_units ])
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

    ic = tf.gather(cate_list, self.i)
    jc = tf.gather(cate_list, self.j)
    hc = tf.gather(cate_list, self.hist_i)

    if use_i_emb:
      i_emb = tf.concat(values = [
          tf.nn.embedding_lookup(item_emb_w, self.i),
          tf.nn.embedding_lookup(cate_emb_w, ic),
          ], axis=1)
      i_b = tf.gather(self.item_b, self.i)
      j_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.j),
        tf.nn.embedding_lookup(cate_emb_w, jc),
      ], axis=1)
      j_b = tf.gather(self.item_b, self.j)
      h_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.hist_i),
        tf.nn.embedding_lookup(cate_emb_w, hc),
      ], axis=2)
    else:
      i_emb = tf.nn.embedding_lookup(cate_emb_w, ic)
      j_emb = tf.nn.embedding_lookup(cate_emb_w, jc)
      h_emb = tf.nn.embedding_lookup(cate_emb_w, hc)

    #-- sum begin -------
    mask = tf.sequence_mask(self.sl, tf.shape(h_emb)[1], dtype=tf.float32) # [B, T]
    mask = tf.expand_dims(mask, -1) # [B, T, 1]
    mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]]) # [B, T, H]
    h_emb *= mask # [B, T, H]
    hist = h_emb
    hist = tf.reduce_sum(hist, 1) 
    hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(self.sl,1), [1, hidden_units]), tf.float32))
    print(h_emb.get_shape().as_list())
    #-- sum end ---------
    
    hist = tf.layers.batch_normalization(inputs = hist, name='bn_hist', training=self.phase)
    hist = tf.reshape(hist, [-1, hidden_units])
    hist = tf.layers.dense(hist, hidden_units)
    hist = tf.layers.dropout(hist, training=self.phase, name='dropout_hist')

    u_emb = hist
    #-- fcn begin -------
    din_i = tf.concat([u_emb, i_emb], axis=-1)
    din_i = tf.layers.batch_normalization(inputs=din_i, name='bn_din', training=self.phase)
    d_layer_1_i = tf.layers.dense(din_i, f1_dim, activation=tf.nn.sigmoid, name='f1')
    d_layer_1_i = tf.layers.dropout(d_layer_1_i, training=self.phase, name='dropout_f1')
    d_layer_2_i = tf.layers.dense(d_layer_1_i, f2_dim, activation=tf.nn.sigmoid, name='f2')
    d_layer_2_i = tf.layers.dropout(d_layer_2_i, training=self.phase, name='dropout_f2')
    d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
    d_layer_3_i = tf.layers.dropout(d_layer_3_i, training=self.phase, name='dropout_f3')
    din_j = tf.concat([u_emb, j_emb], axis=-1)
    din_j = tf.layers.batch_normalization(inputs=din_j, name='bn_din', reuse=True, training=self.phase, trainable=False)
    d_layer_1_j = tf.layers.dense(din_j, f1_dim, activation=tf.nn.sigmoid, name='f1', reuse=True)
    d_layer_2_j = tf.layers.dense(d_layer_1_j, f2_dim, activation=tf.nn.sigmoid, name='f2', reuse=True)
    d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
    d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
    d_layer_3_j = tf.reshape(d_layer_3_j, [-1])

    if use_i_emb:
      x = i_b - j_b + d_layer_3_i - d_layer_3_j # [B]
      self.logits = i_b + d_layer_3_i
    else:
      x = d_layer_3_i - d_layer_3_j # [B]
      self.logits = d_layer_3_i

    u_emb_all = tf.expand_dims(u_emb, 1)
    u_emb_all = tf.tile(u_emb_all, [1, item_count, 1])
    # logits for all item:

    if use_i_emb:
      all_emb = tf.concat([
          item_emb_w,
          tf.nn.embedding_lookup(cate_emb_w, cate_list)
          ], axis=1)
    else:
      all_emb = tf.nn.embedding_lookup(cate_emb_w, cate_list)

    all_emb = tf.expand_dims(all_emb, 0)
    all_emb = tf.tile(all_emb, [B, 1, 1])
    din_all = tf.concat([u_emb_all, all_emb], axis=-1)
    din_all = tf.layers.batch_normalization(inputs=din_all, name='bn_din', reuse=True, training=self.phase, trainable=False)
    d_layer_1_all = tf.layers.dense(din_all, f1_dim, activation=tf.nn.sigmoid, name='f1', reuse=True)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, f2_dim, activation=tf.nn.sigmoid, name='f2', reuse=True)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3', reuse=True)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, item_count])
    if use_i_emb:
      self.logits_all = tf.sigmoid(self.item_b + d_layer_3_all)
    else:
      self.logits_all = tf.sigmoid(d_layer_3_all)
    #-- fcn end -------

    
    self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
    if use_i_emb:
      self.score_i = tf.sigmoid(i_b + d_layer_3_i)
      self.score_j = tf.sigmoid(j_b + d_layer_3_j)
    else:
      self.score_i = tf.sigmoid(d_layer_3_i)
      self.score_j = tf.sigmoid(d_layer_3_j)
    self.score_i = tf.reshape(self.score_i, [-1, 1])
    self.score_j = tf.reshape(self.score_j, [-1, 1])
    self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)
    print(self.p_and_n.get_shape().as_list())


    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = \
        tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = \
        tf.assign(self.global_epoch_step, self.global_epoch_step+1)

    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.y)
        )

    trainable_params = tf.trainable_variables()
    #self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    self.opt = tf.train.AdamOptimizer()
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)

    if use_i_emb:
      tf.summary.histogram('item_b', self.item_b)
    tf.summary.scalar('loss', self.loss)
    self.merged = tf.summary.merge_all()

    # self.train_op = self.opt.apply_gradients(
    #     zip(clip_gradients, trainable_params), global_step=self.global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)



  def train(self, sess, uij, l):
    merged, loss, _ = sess.run([self.merged, self.loss, self.train_op], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.lr: l,
        self.phase: True
        })
    return merged, loss

  def eval(self, sess, uij):
    u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.phase: False
        })
    return u_auc, socre_p_and_n

  def test(self, sess, uid, hist_i, sl):
    return sess.run(self.logits_all, feed_dict={
        self.u: uid,
        self.hist_i: hist_i,
        self.sl: sl,
        self.phase: False
        })

  def prob_item_b(self, sess):
    return sess.run(self.item_b, feed_dict={
        self.phase: False
        })

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)

def extract_axis_1(data, ind):
  batch_range = tf.range(tf.shape(data)[0])
  indices = tf.stack([batch_range, ind], axis=1)
  res = tf.gather_nd(data, indices)
  return res

