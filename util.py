import random
import numpy as np
import tensorflow as tf
    
def get_mnist_batch(batch_size, n_sample, w, h, mnist_data):
    
    """
    Generate a batch on MNIST hand written dataset
    """
    
    batch_index = random.sample(range(n_sample), batch_size)

    batch_data = np.empty([batch_size, w, h, 1], dtype=np.float32)
    for n, i in enumerate(batch_index):
        batch_data[n, ...] = mnist_data[i, ...]

    return batch_data, batch_index

def get_gene_batch_approx_reg(batch_size, n_sample, w, h, gene_data, ppi_mat):
    
    """
    Generate a batch on Visium spatial transcriptomics data 
    and Laplacian matrix on corresponding sub-PPI graph
    """
    
    batch_index = np.array(random.sample(range(n_sample), batch_size))

    batch_data = gene_data[batch_index, ...]

    A = ppi_mat[np.ix_(batch_index, batch_index)]

    d = 1.0/np.sqrt(A.sum(axis=1))
    D_inv = np.diag(np.where(np.isinf(d), 0, d))
    batch_ppi_lap_mat = np.identity(batch_size) - D_inv@A@D_inv

    return batch_data, batch_ppi_lap_mat, batch_index

def get_gene_batch_graph_reg(batch_size, n_sample, w, h, gene_data, ppi_mat):
    
    """
    Generate a batch on Visium spatial transcriptomics data 
    and Laplacian matrix on PPI graph after reordering 
    """
    
    random_index = np.array(random.sample(range(n_sample), n_sample))
    
    batch_index = random_index[0:batch_size]
    rest_index = random_index[batch_size:]

    batch_data = gene_data[batch_index, ...]

    A = ppi_mat[np.ix_(random_index, random_index)]

    d = 1.0/np.sqrt(A.sum(axis=1))
    D_inv = np.diag(np.where(np.isinf(d), 0, d))
    ppi_lap_mat = np.identity(n_sample) - D_inv@A@D_inv

    return batch_data, ppi_lap_mat, batch_index, rest_index

def complete_restore(session, checkpoint_path):
    
    """
    Restore weights for pretrained model on MNIST
    """
    
    saver = tf.train.Saver()
    saver.restore(session, checkpoint_path)
    
    #return saver

def partial_restore(session, checkpoint_path):
    
    """
    Initialize weights for prepended convolution layer and 
    restore weights for pretrained model on MNIST
    """
    
    session.run(tf.global_variables_initializer())
    
    checkpoint_reader = tf.train.NewCheckpointReader(checkpoint_path)
    saved_var_shapes = checkpoint_reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for
                        var in tf.global_variables()
                        if var.name.split(':')[0] in saved_var_shapes])
    
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],
                            tf.global_variables()),
                        tf.global_variables()))
    
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            var = name2var[saved_var_name]
            var_shape = var.get_shape().as_list()
            if var_shape == saved_var_shapes[saved_var_name]:
                restore_vars.append(var)
    
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, checkpoint_path)
    
    #return saver