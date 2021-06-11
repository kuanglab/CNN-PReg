import os
import sys
import cv2
import net
import util
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import mmread

def run_pretraining(args=None):
    
    eps = 1e-10
    base_lr = 0.001 # learning rate
    
    # GPU
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("GPU 0 will be used")
    else:
        sys.exit("Require GPU to run the code")
    
    # Model
    if args.model is not None:
        model_dir = args.model
    else:
        model_dir = os.path.join(os.getcwd(), "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    print("Start pre-training on MNIST data...")
    
    # Load MNIST dataset
    (mnist_train_data, mnist_train_labels), (mnist_test_data, mnist_test_labels) = tf.keras.datasets.mnist.load_data()
    mnist_train_data = np.expand_dims(mnist_train_data, axis=-1)
    mnist_test_data = np.expand_dims(mnist_test_data, axis=-1)

    mnist_data = np.concatenate([mnist_train_data, mnist_test_data], axis=0)
    mnist_labels = np.concatenate([mnist_train_labels, mnist_test_labels], axis=0)
    
    n_sample, w, h, _ = np.shape(mnist_data)
    
    # Normalization
    mnist_data_norm = mnist_data.reshape(mnist_data.shape[0], -1)
    mnist_data_norm = mnist_data_norm/np.amax(mnist_data_norm, axis=1)[:, None]
    mnist_data_norm = mnist_data_norm.reshape(mnist_data.shape)

    imgs = tf.placeholder(shape=[None, w, h, 1], dtype=tf.float32, name='images')
    
    u_thres = tf.placeholder(shape=[], dtype=tf.float32, name='u_thres')
    l_thres = tf.placeholder(shape=[], dtype=tf.float32, name='l_thres')
    lr = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
    
    label_feat = net.mnistNetwork(imgs, args.cluster, name="mnistNetwork", reuse=False)
    label_feat_norm = tf.nn.l2_normalize(label_feat, dim=1)
    # Compute similarity matrix based on embeddings from CNN encoder
    sim_mat = tf.matmul(label_feat_norm, label_feat_norm, transpose_b=True)
    
    pos_loc = tf.greater(sim_mat, u_thres, name='greater')
    neg_loc = tf.less(sim_mat, l_thres, name='less')
    pos_loc_mask = tf.cast(pos_loc, dtype=tf.float32)
    neg_loc_mask = tf.cast(neg_loc, dtype=tf.float32)
    
    pos_entropy = tf.multiply(-tf.log(tf.clip_by_value(sim_mat, eps, 1.0)), pos_loc_mask)
    neg_entropy = tf.multiply(-tf.log(tf.clip_by_value(1-sim_mat, eps, 1.0)), neg_loc_mask)
    
    # Construct loss function based on similarity matrix
    loss_sum = tf.reduce_mean(pos_entropy) + tf.reduce_mean(neg_entropy)

    train_op = tf.train.RMSPropOptimizer(lr).minimize(loss_sum)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        eta = 0 # step size
        epoch = 1
        u = 0.95 # upper threshold
        l = 0.455 # lower threshold
        
        while u > l:

            print("Epoch %d" % epoch)
            
            # Update upper and lower thresholds
            u = 0.95 - eta
            l = 0.455 + 0.1*eta
            
            for i in range(1, int(args.epoch + 1)):
                mnist_batch, batch_index = util.get_mnist_batch(args.batch_size, n_sample, w, h, mnist_data_norm)
                feed_dict={imgs: mnist_batch,
                        u_thres: u,
                        l_thres: l,
                        lr: base_lr}

                train_loss, _ = sess.run([loss_sum, train_op], feed_dict=feed_dict)
                if i % 5 == 0:
                    print('training loss at iter %d is %f' % (i, train_loss))
                    
            # Update step size
            eta += 1.1 * 0.009
            
            # Create checkpoint every 5 epochs 
            if epoch % 5 == 0: 
                model_name = 'CNN_MNIST_ep_' + str(epoch) + '.ckpt'
                save_path = saver.save(sess, os.path.join(model_dir, model_name))
                print("Checkpoint created in file: %s" % save_path)

            epoch += 1

def run_gene_clustering(args=None):
    
    eps = 1e-10
    base_lr = 0.001 # learning rate 
    
    # Extract tissue name
    tn = args.gene_maps.split('/')[-2] if args.gene_maps[-1] is  "/" else args.gene_maps.split('/')[-1]
    
    # GPU
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("GPU 0 will be used")
    else:
        sys.exit("Require GPU to run the code")
    
    # Checkpoint
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        checkpoint = os.path.join(os.getcwd(), "model", "CNN_MNIST_ep_45.ckpt")
        if not os.path.exists(checkpoint):
            sys.exit("Pre-trained model does not exist")
    
    # Model 
    if args.model is not None:
        model_dir = args.model
    else:
        model_dir = os.path.join(os.getcwd(), "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    # Clustering results
    if args.gene_clusters is not None:
        gene_cluster_dir = args.gene_clusters
    else:
        gene_cluster_dir = os.path.join(os.getcwd(), "clustering")
        if not os.path.exists(gene_cluster_dir):
            os.makedirs(gene_cluster_dir)
    
    # Load Mouse or Human PPI graph
    if args.sp:
        print("Load mus musculus PPI network")
        gene_ids = np.loadtxt(args.PPI + '/Mmusculus_gene_list_80.txt', dtype=np.str)
        A = mmread(args.PPI + '/Mmusculus_PPI_80.mtx').toarray()
    else:
        print("Load homo sapiens PPI network")
        gene_ids = np.loadtxt(args.PPI + '/Hsapiens_gene_list_80.txt', dtype=np.str)
        A = mmread(args.PPI + '/Hsapiens_PPI_80.mtx').toarray()
    
    # Load gene activity maps
    gene_data = np.stack([np.load(os.path.join(args.gene_maps, gene_id + '.npy')) for gene_id in gene_ids], axis=0)
    gene_data = np.expand_dims(gene_data, axis=-1)
    # Remove lowly expressed genes 
    gene_filter = np.where(gene_data.reshape(gene_data.shape[0], -1).sum(axis=1) > args.expr_thres)[0]
    gene_ids = gene_ids[gene_filter]
    A = A[np.ix_(gene_filter, gene_filter)]
    
    # Load spatially variable gene list if provided
    if args.svgs is not None:
        print("Start clustering on spatially variable genes for %s ..." % tn)
        svgs = np.loadtxt(args.svgs, dtype=np.str)
        gene_filter = np.where([gene_id in svgs for gene_id in gene_ids])[0]
        gene_ids = gene_ids[gene_filter]
        A = A[np.ix_(gene_filter, gene_filter)]
    else:
        print("Start clustering on all genes for %s ..." % tn)
        
    # Reload gene activity maps
    gene_data = np.stack([np.load(os.path.join(args.gene_maps, 'activity_maps', gene_id + '.npy')) for gene_id in gene_ids], axis=0)
    gene_data_norm = gene_data.reshape(gene_data.shape[0], -1)
    gene_data_norm = gene_data_norm/np.amax(gene_data_norm, axis=1)[:, None]
    gene_data_norm = gene_data_norm.reshape(gene_data.shape)
    
    # Process gene activity maps (padding or resize)
    if args.prep:
        gene_data_norm = np.stack([cv2.resize(gene_data_norm[i, ...], dsize=(28,28)) 
                                   for i in range(gene_data_norm.shape[0])],axis=0)
    else:
        gene_data_norm = np.stack([np.pad(gene_data_norm[i, ...], pad_width=((3, 3), (10, 10))) 
                                   for i in range(gene_data_norm.shape[0])],axis=0)
        
    gene_data_norm = np.expand_dims(gene_data_norm, axis=-1)

    n_gene, w, h, _ = np.shape(gene_data_norm)

    gene_maps = tf.placeholder(shape=[None, w, h, 1], dtype=tf.float32, name='gene_maps')
    lap_mat = tf.placeholder(shape=None, dtype=tf.float32, name='lap_mat')
    
    u_thres = tf.placeholder(shape=[], dtype=tf.float32, name='u_thres')
    l_thres = tf.placeholder(shape=[], dtype=tf.float32, name='l_thres')
    lr = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
    alpha = tf.placeholder(shape=[], dtype=tf.float32, name='alpha')
    
    # Prepend additional convolutional to CNN encoder according to gene activity maps processing 
    if args.prep:
        gene_embs = net.mnistNetwork(gene_maps, args.cluster, name="mnistNetwork", reuse=False)
    else:
        gene_embs = net.VisiumNetwork(gene_maps, args.cluster)
    
    gene_embs_norm = tf.nn.l2_normalize(gene_embs, dim = 1)
    
    # Use exact or approximated PPI graph regularization based on the number of genes involved in the clustering
    if args.svgs is not None:
        # PPI graph regularization
        gene_embs_norm_rest = tf.placeholder(shape=[None, args.cluster], dtype=tf.float32)
        gene_embs_norm_mat = tf.concat([gene_embs_norm, gene_embs_norm_rest], 0)
        # Compute similarity matrix and PPI graph regularization based on all gene embeddings
        sim_mat = tf.matmul(gene_embs_norm_mat, gene_embs_norm_mat, transpose_b=True)
        graph_reg = tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(gene_embs_norm_mat), lap_mat), gene_embs_norm_mat))
        
    else:
        # Approximated PPI graph regularization
        # Compute similarity matrix and PPI graph regularization based on gene embeddings in the batch
        sim_mat = tf.matmul(gene_embs_norm, gene_embs_norm, transpose_b=True)
        graph_reg = tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(gene_embs_norm), lap_mat), gene_embs_norm))
    
    pos_loc = tf.greater(sim_mat, u_thres, name='greater')
    neg_loc = tf.less(sim_mat, l_thres, name='less')
    pos_loc_mask = tf.cast(pos_loc, dtype=tf.float32)
    neg_loc_mask = tf.cast(neg_loc, dtype=tf.float32)
    
    pos_entropy = tf.multiply(-tf.log(tf.clip_by_value(sim_mat, eps, 1.0)), pos_loc_mask)
    neg_entropy = tf.multiply(-tf.log(tf.clip_by_value(1-sim_mat, eps, 1.0)), neg_loc_mask)
    
    graph_reg = tf.math.divide(graph_reg, args.cluster)
    
    # Construct combined loss (clustering loss and PPI graph regularization)
    loss_sum = tf.reduce_mean(pos_entropy) + tf.reduce_mean(neg_entropy) + tf.multiply(graph_reg, alpha)

    train_op = tf.train.RMSPropOptimizer(lr).minimize(loss_sum)
    
    # Infer gene cluster membership based on gene embeddings
    gene_clusters = tf.argmax(gene_embs, axis=1)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        # Load pre-trained CNN
        if args.prep:
            util.complete_restore(sess, checkpoint)
        else:
            util.partial_restore(sess, checkpoint)
        
        print('Pre-trained model restored!')

        eta = 0 # step size
        epoch = 1
        u = 0.95 # threshold for similar gene selection
        l = 0.455 # threshold for dissimilar gene selection
        
        # Create gene embedding matrix when fewer genes involved in the clustering
        if args.svgs is not None:
            
            F = np.zeros((n_gene, args.cluster))
            for j in range(int(np.ceil(n_gene/args.batch_size))):
                gene_batch = np.copy(gene_data_norm[args.batch_size*j:args.batch_size*(j+1), ...])
                feed_dict={gene_maps: gene_batch}
                F[j*args.batch_size:(j+1)*args.batch_size, ...] = sess.run(gene_embs_norm, feed_dict=feed_dict)

        while u > l:

            print("Epoch %d" % epoch)
            
            # Update thresholds for both similar and dissimilar gene selection
            u = 0.95 - eta
            l = 0.455 + 0.1*eta

            for i in range(1, int(args.epoch + 1)):
                
                if args.svgs is not None:
                    
                    gene_batch, ppi_lap_mat, batch_index, rest_index = util.get_gene_batch_graph_reg(args.batch_size, n_gene,
                                                                                                     w, h, gene_data_norm, A)

                    feed_dict={gene_maps: gene_batch,
                               gene_embs_norm_rest: F[rest_index, ...],
                               lap_mat: ppi_lap_mat,
                               alpha: args.alpha,
                               u_thres: u,
                               l_thres: l,
                               lr: base_lr}
                    
                else:
                    
                    gene_batch, ppi_lap_mat, batch_index = util.get_gene_batch_approx_reg(args.batch_size, n_gene, 
                                                                                          w, h, gene_data_norm, A)

                    feed_dict={gene_maps: gene_batch,
                            lap_mat: ppi_lap_mat,
                            alpha: args.alpha,
                            u_thres: u,
                            l_thres: l,
                            lr: base_lr}

                train_loss, _ = sess.run([loss_sum, train_op], feed_dict=feed_dict)
                
                # Update gene embedding matrix when fewer genes involved in the clustering
                if args.svgs is not None:
                    
                    for j in range(int(np.ceil(n_gene/args.batch_size))):
                        gene_batch = np.copy(gene_data_norm[args.batch_size*j:args.batch_size*(j+1), ...])
                        feed_dict={gene_maps: gene_batch}
                        F[j*args.batch_size:(j+1)*args.batch_size, ...] = sess.run(gene_embs_norm, feed_dict=feed_dict)

                if i % 20 == 0:
                    print('training loss at iter %d is %f' % (i, train_loss))
                    
            # Update step size
            eta += 1.1 * 0.009
            
            # Create checkpoint every 5 epochs
            if epoch % 5 == 0:  # save model at every 5 epochs
                model_name = 'CNN_PReg_ep_' + str(epoch) + '.ckpt'
                save_path = saver.save(sess, os.path.join(model_dir, model_name))
                print("Checkpoint created in file: %s" % save_path)

            epoch += 1

    with tf.Session() as sess:
        
        # Load the most recent checkpoint
        saver.restore(sess, os.path.join(model_dir, 'CNN_PReg_ep_45.ckpt'))
        
        # Infer gene memberships
        all_gene_clusters = np.zeros([n_gene], dtype=np.float32)
        for j in range(int(np.ceil(n_gene/args.batch_size))):
            gene_batch = np.copy(gene_data_norm[args.batch_size*j:args.batch_size*(j+1), ...])
            feed_dict={gene_maps: gene_batch}
            all_gene_clusters[j*args.batch_size:(j+1)*args.batch_size] = sess.run(gene_clusters, feed_dict=feed_dict)

    data = pd.DataFrame({'gene_id': gene_ids, 'cluster_id': all_gene_clusters}, columns=['gene_id', 'cluster_id'])
    data.to_csv(os.path.join(gene_cluster_dir,  tn + '_gene_clusters.csv'), index=False)