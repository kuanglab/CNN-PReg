#!/usr/bin/env python

# Before running this script to generate gene activity maps, please make sure you have already 
# downloaded Visium data from 10x genomics (Space Ranger v1.0.0):
#
#       https://support.10xgenomics.com/spatial-gene-expression/datasets/
#
# Available tissues on 10x genomics:
#
#       V1_Adult_Mouse_Brain                              V1_Breast_Cancer_Block_A_Section_1
#       V1_Breast_Cancer_Block_A_Section_2                V1_Human_Heart
#       V1_Human_Lymph_Node                               V1_Mouse_Brain_Sagittal_Anterior
#       V1_Mouse_Brain_Sagittal_Anterior_Section_2        V1_Mouse_Brain_Sagittal_Posterior
#       V1_Mouse_Brain_Sagittal_Posterior_Section_2       V1_Mouse_Kidney
#
#
# Note that we only use filtered feature-barcode matrix data and spatial coordindates, and they 
# can be downloaded with following links (replace "tissue-name" with one from above list)
#
#       filtered feature-barcode matrix data: 
#       https://support.10xgenomics.com/spatial-gene-expression/datasets/<tissue-name>/<tissue-name>_filtered_feature_bc_matrix.tar.gz
#
#       spatial coordinates:
#       https://support.10xgenomics.com/spatial-gene-expression/datasets/<tissue-name>/<tissue-name>_spatial.tar.gz
#
# And then unzip the downloaded data and organize folders using the following structure:
#
#       . <data-folder>
#       ├── ...
#       ├── <tissue-folder>
#       │   ├── filtered_feature_bc_matrix
#       │   │   ├── barcodes.tsv.gz
#       │   │   ├── features.tsv.gz
#       │   │   └── matrix.mtx.gz 
#       │   ├── spatial
#       │   │   ├── tissue_positions_list.csv
#       └── ...
#
# To learn more about the filtered feature-barcode matrix data and spatial coordinates, 
# please visit the links below:
#
#       filtered feature-barcode matrix data: 
#       https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/matrices
#
#       spatial information:
#       https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/images
#

import os
import csv
import gzip
import argparse
import numpy as np
from tqdm import tqdm
from scipy.io import mmread
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed

# arguments parser
parser = argparse.ArgumentParser('Preprocessing')
parser.add_argument("--raw", type=str, default=None, help="path to raw data folder")
parser.add_argument("--gene_maps", type=str, default=None, help="path to gene activity maps folder")
parser.add_argument("--job", type=int, default=4, help="number of jobs")

args = parser.parse_args()

print("Start preprocessing ...")

# List all tissues for convertion
tissue_names = np.sort([dir_name for dir_name in os.listdir(args.raw) if os.path.isdir(os.path.join(args.raw, dir_name))])

for tn in tissue_names[21:22]:
    
    print("Tissue name: %s" % tn)

    # Set path to filtered feature-barcode matrix data and spatial coordinates
    matrix_dir = os.path.join(args.raw, tn, "filtered_feature_bc_matrix")
    features_path = os.path.join(matrix_dir, "features.tsv.gz")
    barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
    sp_info_dir = os.path.join(args.raw, tn, "spatial")
    sp_info_path = os.path.join(sp_info_dir, "tissue_positions_list.csv")
    
    if args.gene_maps is None:
        gene_maps_dir = os.path.join(args.raw, tn, "activity_maps")
    else:
        gene_maps_dir = os.path.join(args.gene_maps, tn, "activity_maps")
    
    if not os.path.exists(gene_maps_dir):
        os.makedirs(gene_maps_dir)
    print("Gene activity maps will be generated under %s" % gene_maps_dir)

    # Read filtered feature-barcode matrix data
    mat = mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))
    feature_ids = np.array([row[0] for row in csv.reader(gzip.open(features_path, 'rt'), delimiter="\t")])
    gene_names = np.array([row[1] for row in csv.reader(gzip.open(features_path, 'rt'), delimiter="\t")])
    feature_types = np.array([row[2] for row in csv.reader(gzip.open(features_path, 'rt'), delimiter="\t")])
    barcodes = np.array([row[0] for row in csv.reader(gzip.open(barcodes_path, 'rt'), delimiter="\t")])

    # Read spatial coordinates
    sp_barcodes = np.array([row[0] for row in csv.reader(open(sp_info_path, 'rt'), delimiter=",")])
    sp_x_coords = np.array([row[2] for row in csv.reader(open(sp_info_path, 'rt'), delimiter=",")], dtype=np.int32)
    sp_y_coords = np.array([row[3] for row in csv.reader(open(sp_info_path, 'rt'), delimiter=",")], dtype=np.int32)

    # Align spatial coordinates between rows
    idx = [np.where(np.array(sp_barcodes) == barcode)[0][0] for barcode in barcodes]
    x_coords = sp_x_coords[idx]
    y_coords = sp_y_coords[idx]
    x_aligned_coords = x_coords
    y_aligned_coords = y_coords//2
    
    # Generate gene activity maps
    gene_data = np.stack([coo_matrix((mat.data[np.where(mat.row==row)[0]], 
                                      (x_aligned_coords[mat.col[np.where(mat.row==row)[0]]], 
                                       y_aligned_coords[mat.col[np.where(mat.row==row)[0]]])), 
                                     shape=(78, 64)).toarray() 
                          for row in np.unique(mat.row)], axis = 0)
    gene_ids = feature_ids[np.unique(mat.row)]

    Parallel(n_jobs=args.job)(delayed(np.save)(os.path.join(gene_maps_dir, feature_id + '.npy'), np.zeros((78, 64))) for feature_id in feature_ids if feature_id not in gene_ids)
    
    Parallel(n_jobs=args.job)(delayed(np.save)(os.path.join(gene_maps_dir, gene_ids[i] + '.npy'), gene_data[i, ...]) for i in tqdm(range(len(gene_ids))))
    
    print("Complete!")