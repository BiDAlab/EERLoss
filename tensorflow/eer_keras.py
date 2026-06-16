import  os;
import  tensorflow_addons   as tfa;

import  sys;
import  tensorflow                  as      tf;
from    tensorflow.keras.losses     import  Loss;
from    tensorflow_addons.losses    import  metric_learning;



@tf.function
def binary_search_deer(legitimate_scores, impostor_scores, lr):
  K = tf.cast(100.0, dtype=tf.float32);
  d = tf.reduce_mean(lr);

  fn  = tf.math.tanh(K * (tf.math.maximum(d, legitimate_scores) - d));
  frr = 100.0 * tf.reduce_mean(fn);

  fp  = tf.math.tanh(K * (d - tf.math.minimum(d, impostor_scores)));
  far = 100.0 * tf.reduce_mean(fp);

  m = tf.maximum(frr, far);
  cl = tf.math.exp(K * (frr - m));
  cr = tf.math.exp(K * (far - m));

  nl = lr[0] * (1.0 - cl) + d * cl;
  nr = lr[1] * (1.0 - cr) + d * cr

  # tf.print(tf.strings.format("[eer-binary]    l={}    r={}    d={}    far={}    frr={}    cl={}    cr={}    nl={}    nr={}", [lr[0], lr[1], d, far, frr, cl, cr, nl, nr]), output_stream=sys.stdout);
  return tf.stack([nl, nr]);



@tf.function
def calculate_eer_loss(y_true, y_pred, total_sets, binary_search_steps, legitimate_mask, impostor_mask) -> tf.Tensor:
    embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
    );
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
    );

    pdist_matrix = metric_learning.pairwise_distance(
        precise_embeddings, squared=False
    );
    
    legitimate_scores = tf.boolean_mask(pdist_matrix, legitimate_mask);
    impostor_scores = tf.boolean_mask(pdist_matrix, impostor_mask);
    
    uL = tf.reduce_mean(legitimate_scores);
    uI = tf.reduce_mean(impostor_scores);
    # tf.print(tf.strings.format("[eer-binary]    uL={}    uI={}", [uL, uI]), output_stream=sys.stdout);

    lr = tf.stack([0.5 * uL, 1.5 * uI]);
    for i in range(binary_search_steps):
       lr = binary_search_deer(legitimate_scores, impostor_scores, lr);

    deer = tf.reduce_mean(lr);
    
    K = tf.cast(100.0, dtype=tf.float32);
    fn  = tf.math.tanh(K * (tf.math.maximum(deer, legitimate_scores) - deer));
    frr = 100.0 * tf.reduce_mean(fn);

    fp  = tf.math.tanh(K * (deer - tf.math.minimum(deer, impostor_scores)));
    far = 100.0 * tf.reduce_mean(fp);

    retval = (far + frr) / 2.0;
    tf.print(tf.strings.format("[eer-keras] deer={}    loss={}", [deer, retval]), output_stream=sys.stdout);
    return retval;





class EERLoss(Loss):
    def __init__(self, sets_per_batch, samples_per_set, binary_search_steps):
        super(EERLoss, self).__init__()
        self.sets_per_batch = sets_per_batch;
        self.samples_per_set = samples_per_set;
        self.binary_search_steps = binary_search_steps;

        LEN = sets_per_batch * samples_per_set;
        legitimate_mask = tf.zeros([LEN,LEN], dtype=float)
        impostor_mask   = tf.ones([LEN,LEN], dtype=float)
        impostor_mask   = tf.linalg.band_part(impostor_mask, -1, 0) - tf.linalg.band_part(impostor_mask, 0, 0);

        scalar_one = 1.0
        vector_one = tf.constant([scalar_one]);
        scalar_zero = 0.0;
        vector_zero = tf.constant([scalar_zero]);

        for i in range(0,self.sets_per_batch):
          for j in range(0,samples_per_set):
            for k in range(0,j):
              position = [samples_per_set * i + j, samples_per_set * i + k];
              index = tf.constant([position])
              legitimate_mask = tf.tensor_scatter_nd_update(legitimate_mask, index, vector_one);
              impostor_mask = tf.tensor_scatter_nd_update(impostor_mask, index, vector_zero);

        self.legitimate_mask = legitimate_mask;
        self.impostor_mask = impostor_mask;


    def call(self, y_true, y_pred):
        return calculate_eer_loss(y_true, y_pred, self.sets_per_batch, self.binary_search_steps, self.legitimate_mask, self.impostor_mask);



def get_loss(sets_per_batch, samples_per_set, binary_search_steps):
    return EERLoss(sets_per_batch, samples_per_set, binary_search_steps);





