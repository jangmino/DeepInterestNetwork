import tensorflow as tf

from Dice import dice

class Model(object):

  def __init__(self, user_count, item_count, cate_count, cate_list, use_dice=False):

    self.u = tf.placeholder(tf.int32, [None,], name="ph_u") # [B]
    self.i = tf.placeholder(tf.int32, [None,], name="ph_i") # [B]
    self.j = tf.placeholder(tf.int32, [None,], name="ph_j") # [B]
    self.y = tf.placeholder(tf.float32, [None,], name="ph_y") # [B]
    self.hist_i = tf.placeholder(tf.int32, [None, None], name="ph_hist_i") # [B, T]
    self.sl = tf.placeholder(tf.int32, [None,], name="ph_sl") # [B]
    self.lr = tf.placeholder(tf.float64, [], name="ph_lr")
    self.phase = tf.placeholder(tf.bool, name="pholder_phase")

    hidden_units = 128

    B = tf.shape(self.u)[0]

    user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
    item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
    item_b = tf.get_variable("item_b", [item_count],
                             initializer=tf.constant_initializer(0.0))
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

    u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)

    ic = tf.gather(cate_list, self.i)
    i_emb = tf.concat(values = [
        tf.nn.embedding_lookup(item_emb_w, self.i),
        tf.nn.embedding_lookup(cate_emb_w, ic),
        ], axis=1)
    i_b = tf.gather(item_b, self.i)

    jc = tf.gather(cate_list, self.j)
    j_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.j),
        tf.nn.embedding_lookup(cate_emb_w, jc),
        ], axis=1)
    j_b = tf.gather(item_b, self.j)

    hc = tf.gather(cate_list, self.hist_i)
    h_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.hist_i),
        tf.nn.embedding_lookup(cate_emb_w, hc),
        ], axis=2)

    hist = attention(i_emb, h_emb, self.sl, reuse=False) # [B, 1, H]
    #-- attention end ---

    hist = tf.layers.batch_normalization(inputs=hist, name='bn_hist', training=self.phase)
    hist = tf.reshape(hist, [-1, hidden_units])
    hist = tf.layers.dense(hist, hidden_units, name='dense_hist')

    u_emb = hist
    print(u_emb.get_shape().as_list())
    print(i_emb.get_shape().as_list())
    print(j_emb.get_shape().as_list())
    #-- fcn begin -------
    din_i = tf.concat([u_emb, i_emb], axis=-1)
    din_i = tf.layers.batch_normalization(inputs=din_i, name='bn_din', training=self.phase)
    if use_dice:
      d_layer_1_i = tf.layers.dense(din_i, 80, activation=None, name='f1')
      d_layer_1_i = dice(d_layer_1_i, name='dice_1_i', training=self.phase, trainable=True)
      d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=None, name='f2')
      d_layer_2_i = dice(d_layer_2_i, name='dice_2_i', training=self.phase, trainable=True)
    else:
      d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
      d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')

    d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')

    hist_j = attention(j_emb, h_emb, self.sl, reuse=True)
    hist_j = tf.layers.batch_normalization(inputs=hist_j, name='bn_hist', reuse=True, training=self.phase, trainable=False)
    hist_j = tf.reshape(hist_j, [-1, hidden_units])
    hist_j = tf.layers.dense(hist_j, hidden_units, name='dense_hist', reuse=True)

    uj_emb = hist_j

    din_j = tf.concat([uj_emb, j_emb], axis=-1)
    din_j = tf.layers.batch_normalization(inputs=din_j, name='bn_din', reuse=True, training=self.phase, trainable=False)
    if use_dice:
      d_layer_1_j = tf.layers.dense(din_j, 80, activation=None, name='f1', reuse=True)
      d_layer_1_j = dice(d_layer_1_j, name='dice_1_i', training=self.phase, trainable=False, reuse=True)
      d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=None, name='f2', reuse=True)
      d_layer_2_j = dice(d_layer_2_j, name='dice_2_i', training=self.phase, trainable=False, reuse=True)
    else:
      d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
      d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)

    d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
    d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
    d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
    x = i_b - j_b + d_layer_3_i - d_layer_3_j # [B]
    self.logits = i_b + d_layer_3_i

    # predicting for each user over all items.
    all_emb = tf.concat([
        item_emb_w,
        tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)
    all_emb = tf.tile(all_emb, [B, 1]) # [B*I, H]

    h_emb_all = tf.expand_dims(h_emb, 1)
    h_emb_all = tf.tile(h_emb_all, [1, item_count, 1, 1])
    h_emb_all = tf.reshape(h_emb_all, [-1, tf.shape(h_emb)[1], hidden_units]) #[B*I, T, H]

    sl_all = tf.expand_dims(self.sl, 1)
    sl_all = tf.tile(sl_all, [1, item_count])
    sl_all = tf.reshape(sl_all, [-1]) #[B*I,]

    hist_all = attention(all_emb, h_emb_all, sl_all, reuse=True)
    hist_all = tf.layers.batch_normalization(inputs=hist_all, name='bn_hist', reuse=True, training=self.phase, trainable=False)
    hist_all = tf.reshape(hist_all, [-1, hidden_units])
    hist_all = tf.layers.dense(hist_all, hidden_units, name='dense_hist', reuse=True)

    uall_emb = hist_all
    din_all = tf.concat([uall_emb, all_emb], axis=-1)
    din_all = tf.layers.batch_normalization(inputs=din_all, name='bn_din', reuse=True, training=self.phase,
                                            trainable=False)
    if use_dice:
      d_layer_1_all = tf.layers.dense(din_all, 80, activation=None, name='f1', reuse=True)
      d_layer_1_all = dice(d_layer_1_all, name='dice_1_i', training=self.phase, trainable=False, reuse=True)
      d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=None, name='f2', reuse=True)
      d_layer_2_all = dice(d_layer_2_all, name='dice_2_i', training=self.phase, trainable=False, reuse=True)
    else:
      d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
      d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)

    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3', reuse=True)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, item_count]) # [B, I]
    self.logits_all = tf.sigmoid(item_b + d_layer_3_all)
    #-- fcn end -------

    self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
    self.score_i = tf.sigmoid(i_b + d_layer_3_i)
    self.score_j = tf.sigmoid(j_b + d_layer_3_j)
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

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)

  def train(self, sess, uij, l):
    loss, _ = sess.run([self.loss, self.train_op], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.lr: l,
        self.phase: True
        })
    return loss

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

def attention(queries, keys, keys_length, reuse=False):
  '''
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
  '''
  queries_hidden_units = queries.get_shape().as_list()[-1]
  queries = tf.tile(queries, [1, tf.shape(keys)[1]])
  queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=reuse)
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=reuse)
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=reuse)
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
  outputs = d_layer_3_all 
  # Mask
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, 1, T]

  # Weighted sum
  outputs = tf.matmul(outputs, keys)  # [B, 1, H]

  return outputs

