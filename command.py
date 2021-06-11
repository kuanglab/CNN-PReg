import argparse

def parser():

    parser = argparse.ArgumentParser('CNN_PReg')
    
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--gpu", action="store_const", const=True, dest="gpu", default=True, help="GPU")
    gpu.add_argument("--cpu", action="store_const", const=False, dest="gpu", default=False, help="CPU")
    
    subparsers = parser.add_subparsers(dest = 'task')
    
    p_parser = subparsers.add_parser("pretraining", help="pre-train CNN model on MNIST")
    p_parser.add_argument("--model", type=str, default=None, help="root path to save model checkpoints")
    p_parser.add_argument("--cluster", type=int, default=100, help="number of clusters")
    p_parser.add_argument("--batch_size", type=int, default=128, help="training batch size")
    p_parser.add_argument("--epoch", type=int, default=1000, help="number of epochs")
    
    gc_parser = subparsers.add_parser("gene_clustering", help="perform clustering on gene activity maps")
    gc_parser.add_argument("--gene_maps", type=str, default=None, help="root path to load gene activity maps")
    gc_parser.add_argument("--PPI", type=str, default=None, help="root path to load PPI graph")
    
    species = gc_parser.add_mutually_exclusive_group()
    species.add_argument("--Mmusculus", action="store_const", const=True, dest="sp", default=True, help="Mouse")
    species.add_argument("--Hsapiens", action="store_const", const=False, dest="sp", default=False, help="Human")
    
    gc_parser.add_argument("--svgs", type=str, default=None, help="path to spatially variable gene list")
    gc_parser.add_argument("--checkpoint", type=str, default=None, help="path to the checkpoint for pre-trained model")
    
    gc_parser.add_argument("--cluster", type=int, default=100, help="number of clusters")
    gc_parser.add_argument("--batch_size", type=int, default=128, help="training batch size")
    gc_parser.add_argument("--epoch", type=int, default=1000, help="number of epochs")
    gc_parser.add_argument("--alpha", type=float, default=0.01, help="weight on PPI graph regularization")
    
    gc_parser.add_argument("--expr_thres", type=int, default=100, help="expression threshold for genes")
    preprocess = gc_parser.add_mutually_exclusive_group()
    preprocess.add_argument("--resize", action="store_const", const=True, dest="prep", default=True, help="resize")
    preprocess.add_argument("--project", action="store_const", const=False, dest="prep", default=False, help="project")
    
    gc_parser.add_argument("--model", type=str, default=None, help="root path to save model checkpoints")
    gc_parser.add_argument("--gene_clusters", type=str, default=None, help="root path to save gene clustering results")
    
    return parser

