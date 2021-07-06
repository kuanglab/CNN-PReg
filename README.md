Detecting Spatially Co-expressed Gene Clusters with Functional Coherence by Graph-regularized Convolutional Neural Network
--------------------------------------------------------------------------------


Package requirements
--------------------------------------------------------------------
```
[python 3.6.7]
[TensorFlow 1.12.0]
[numpy 1.18.1]
[scipy 1.3.2]
[opencv 4.2.0]
[pandas 1.0.1]
[joblib 0.11]
[tqdm 4.15.0]
```

Data preparation
--------------------------------------------------------------------------------

#### Spatial Transcriptomics Data
Download Visium spatial transcriptomics data from [10x Genomics](https://support.10xgenomics.com/spatial-gene-expression/datasets/) or [spatialLIBD](http://research.libd.org/spatialLIBD/) and make sure these data are organized in the following structure:

        . <data-folder>
        ├── ...
        ├── <tissue-folder>
        │   ├── filtered_feature_bc_matrix
        │   │   ├── barcodes.tsv.gz
        │   │   ├── features.tsv.gz
        │   │   └── matrix.mtx.gz
        │   ├── spatial
        │   │   └── tissue_positions_list.csv
        └── ...

After placing data as specified, run the Python script `preprocessing.py` to generate gene activity maps as follows (Note that if the path to output folder is not provided, a folder named `activity_maps` will be automatically created under the corresponding tissue folder):
```
python3 preprocessing.py --raw <raw_data_folder> --gene_maps <gene_maps_folder>
```

* `--raw`: the path to downloaded data
* `--gene_maps`: the path to gene activity maps folder

#### Protein-Protein Interaction (PPI) Networks
Both `Mus musculus` and `Homo sapiens` PPI networks are obtained from [StringDB](https://string-db.org/), we retained protein-protein interactions with confidence scores larger than `0.8` in this study, PPI network can be found under the folder `data`, where `<species>_PPI_80.mtx` and  `<species>_gene_list_80.txt` contains adjacency matrix and ensembl gene ID list for corresponding PPI network respectively.


#### Spatially Variable Genes (Optional)
In this study, spatially variable gene are commonly confirmed by 6 popular spatially variable gene identification algorithms, including [trendSceek](https://github.com/edsgard/trendsceek), [SpatialDE](https://github.com/Teichlab/SpatialDE), [SPARK](https://github.com/xzhoulab/SPARK), [spatial auto-correlation](https://github.com/jbergenstrahle/STUtility), [binSpect](https://github.com/RubD/Giotto) and [SilhouetteRank](https://github.com/RubD/Giotto). To reproduce the results shown in the paper, `<tissue-name>_top_2000_svgs.txt` can be found under the folder `data`


Pre-training
--------------------------------------------------------------------------------
Run `main.py pretraining` to pre-train the model on MNIST hand written as follows (Note that you can use provided pre-trained model in the `model` folder):

```
python3 main.py pretraining --epoch <no_epochs> --cluster <no_clusters> --model <model_path>
```

* `--epoch`: the number of epochs for training
* `--cluster`: the number of clusters
* `--model`: the path to store model checkpoints folder

Gene clustering
--------------------------------------------------------------------------------
Once completing pre-training, run `main.py gene_clustering` again to perform gene clustering on Visium spatial transcriptomics data:

```
# Run clustering on spatially variable genes
python3 main.py gene_clustering --epoch <no_epochs> --cluster <no_clusters> --alpha <alpha> --gene_maps <gene_maps_folder> --PPI <PPI_folder> --svgs <svg_path> --model <model_path> --checkpoint <pretrained_model_path> --gene_clusters <result_path>
```
```
# Run clustering on all genes
python3 main.py gene_clustering --epoch <no_epochs> --cluster <no_clusters> --alpha <alpha> --gene_maps <gene_maps_folder> --PPI <PPI_folder> --model <model_path> --checkpoint <pretrained_model_path> --gene_clusters <result_path>
```

* `--epoch`: the number of epochs for training
* `--cluster`: the number of clusters
* `--alpha`: the weight on PPI graph regularization
* `--gene_maps`: the path to gene activity maps folder
* `--PPI`: the path to PPI network folder
* `--model`: the path to store model checkpoints folder
* `--checkpoint`: the path to specific pre-trained model checkpoint
* `--model`: the path to model checkpoints folder
* `--gene_clusters`: the path to gene clustering results

References
--------------------------------------------------------------------------------
(Under review)
