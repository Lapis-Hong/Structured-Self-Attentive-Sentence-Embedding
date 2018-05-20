import tensorflow as tf


class SelfAttentive(object):
    """A Structured Self Attentive Sentence Embedding"""
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 sequence_length,
                 hidden_units,
                 num_layers,
                 d_a=350,
                 r=30,
                 dropout_keep_prob=0.5,
                 ):
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length])
        batch_size = tf.shape(self.input_x)[0]
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("embedding"):
            embed_W = tf.get_variable('embedding', shape=[vocab_size, embedding_dim],
                                      initializer=initializer)
        self.input_embed = tf.nn.embedding_lookup(embed_W, self.input_x)

        with tf.variable_scope("BiLSTM"):
            with tf.variable_scope("fw"):
                fw_cells = [tf.nn.rnn_cell.LSTMCell(hidden_units) for _ in range(num_layers)]
                fw_cells = tf.nn.rnn_cell.MultiRNNCell(cells=fw_cells, state_is_tuple=True)
                fw_cells = tf.nn.rnn_cell.DropoutWrapper(fw_cells, output_keep_prob=dropout_keep_prob)
            with tf.variable_scope("bw"):
                bw_cells = [tf.nn.rnn_cell.LSTMCell(hidden_units) for _ in range(num_layers)]
                bw_cells = tf.nn.rnn_cell.MultiRNNCell(cells=bw_cells, state_is_tuple=True)
                bw_cells = tf.nn.rnn_cell.DropoutWrapper(bw_cells, output_keep_prob=dropout_keep_prob)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cells, bw_cells, self.input_embed, dtype=tf.float32)
            # H (h1, h2, .... hn)  shape(H) = n * 2u, n is sequence length, u is hidden units
            self.H = tf.concat([outputs[0], outputs[1]], axis=2, name="H")

        with tf.variable_scope("attention"):
            # shape(W_s1) = d_a * 2u
            self.W_s1 = tf.get_variable('W_s1', shape=[d_a, 2 * hidden_units], initializer=initializer)
            # shape(W_s2) = r * d_a
            self.W_s2 = tf.get_variable('W_s2', shape=[r, d_a], initializer=initializer)
            # shape (d_a, 2u) --> shape(batch_size, d_a, 2u)
            self.W_s1 = tf.tile(tf.expand_dims(self.W_s1, 0), [batch_size, 1, 1])
            self.W_s2 = tf.tile(tf.expand_dims(self.W_s2, 0), [batch_size, 1, 1])
            # attention matrix A = softmax(W_s2*tanh(W_s1*H^T)  shape(A) = r * n  note: first dim is batch_size
            self.H_T = tf.transpose(self.H, perm=[0, 2, 1], name="H_T")
            self.A = tf.nn.softmax(tf.matmul(self.W_s2, tf.tanh(tf.matmul(self.W_s1, self.H_T)), name="A"))
            # sentences embedding matrix M = AH  shape(M) = (batch_size, r, 2u)
            self.M = tf.matmul(self.A, self.H, name="M")

        with tf.variable_scope("penalization"):
            # penalization term: Frobenius norm square of matrix AA^T-I, ie. P = |AA^T-I|_F^2
            A_T = tf.transpose(self.A, perm=[0, 2, 1], name="A_T")
            I = tf.eye(r, r, batch_shape=[batch_size], name="I")
            self.P = tf.square(tf.norm(tf.matmul(self.A, A_T) - I, axis=[-2, -1], ord='fro'), name="P")

    @property
    def varibales(self):
        return tf.global_variables()


if __name__ == '__main__':
    model = SelfAttentive(vocab_size=10000,
                          embedding_dim=128,
                          sequence_length=15,
                          hidden_units=100,
                          num_layers=2,
                          d_a=350,
                          r=30,
                          dropout_keep_prob=0.5,
                          )
    vars = model.varibales
    for var in vars:
        print(var)