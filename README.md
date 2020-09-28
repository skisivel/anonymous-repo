### Container configuration
To run in docker container:
1. Run `docker build -t local-torch-geometric .` in `torch-geometric-docker`.
1. Run `docker build -t gcn-split .` in root folder.
1. If you want to store the data after the container is closed run `docker volume create gcn-split-vol`
1. To start the container run: `docker run -dit --mount source=gcn-split-vol,target=/app --shm-size 8G --entrypoint=/bin/bash gcn-split`.

Then inside the container you are able to run the programs manually 
providing proper arguments.

In case of `Bus error.` problem try setting `shm-size` (shared memory) parameter of the container to a higher value.

If you want to run the streaming experiments (`run_stream_partitioning.py`) on multi-core CPU it is necessary to set following environmental variables inside the container:

`export MKL_NUM_THREADS=1`

`export NUMEXPR_NUM_THREADS=1`

`export OMP_NUM_THREADS=1`

### Commands examples
The following instructions must be carried out from within the container.

To enter the running container run:
`docker exec -it --user=root CONTAINER_ID bash`

`CONTAINER_ID` can be retrieved from `docker ps` command.

#### To run the training of the unsupervised model on Twitch dataset:
1. Make sure that `twitch/DE/musae_DE_edges.csv` and `twitch/DE/musae_DE_features.json` exist.
1. Run `python src/main.py --epochs 3 --b_sz 20 --cut_coeff 1 --bal_coeff 0.00001 --learn_method gap --dataset twitch --agg_func MAX --num_classes 6`
1. After successful run the program should create two files - one containing full model and the other one containing GraphSAGE model only.

 
#### To run the training of the supervised model on Twitch dataset:
1. Make sure that `twitch/DE/musae_DE_edges.csv`, `twitch/DE/musae_DE_features.json` and `twitch/DE/musae_DE_edges.csv_labeled_6` exist.
1. Run `python src/main.py --epochs 0 --sup_epochs 15 --sup_b_sz 2000 --cut_coeff 1 --bal_coeff 0 --learn_method sup_edge --dataset twitch --agg_func MAX --num_classes 6`
1. After successful run the program should create two files - one containing full model and the other one containing GraphSAGE model only.

The names of the files will be similar to: `ds-twitch_gap_mb-20_e-3_se-200_smb-0_cut-1.0_bal-1e-05_agg-MAX-num_classes-6-bfs-0-lr-7.5e-05-Sep17_17-09-00.dot.torch`,
the one containing GraphSAGE model only has `.GRAPHSAGE` postfix.

#### To run the streaming partitioning using pre-trained unsupervised model:
1. Make sure that `./models/twitch-unsup/twitch-unsup-6-partitions.torch`, `./twitch/ENGB/musae_ENGB_edges.csv` and `./twitch/ENGB/musae_ENGB_features.json` exist.
1. Run `python src/run_stream_partitioning.py --learn_method gap --dataset twitch --num_classes 6 --max_load 1.01 
--model ./models/twitch-unsup/twitch-unsup-6-partitions.torch --inf_b_sz 1000 --num_processes 1 
--edge_file_path ./twitch/ENGB/musae_ENGB_edges.csv --feats_file_path ./twitch/ENGB/musae_ENGB_features.json`
1. This will produce the results file with the name based on partitioning and model configuration, e.g.: 
`ds-twitch-1000_win-size-twitch-unsup-6-partitions.torch_6_Sep16_18-34-36_RESULTS.csv`

#### To evaluate the partitioning results:
1. Run `python src/run_evaluate.py --input_file "../ds-twitch-1000_win-size-twitch-unsup-6-partitions.torch_6_Sep16_18-34-36_RESULTS.csv" --num_classes 6`

### Detailed instructions on each command's arguments

#### main.py - training
`--dataset` - dataset name, possible options: `reddit`, `twitch`, `papers100m`, `deezer`, `bitcoin`

`--agg_func=MAX` - aggregation function used by GCN; `MAX` or `MEAN`

`--epochs` - number of epochs of unsupervised (gap) training

`--sup_epochs` - number of epochs of supervised training

`--b_sz` - batch size (number of nodes) in unsupervised training

`--sup_b_sz` - batch size (number of edges) in supervised training

`--cut_coeff` - coefficient of the min-cut part of the loss function

`--bal_coeff` - coefficient of the balancing part of the loss function

`--num_classes` - number of partitions

`--bfs` - whether to use BFS algorithm for neighbourhood sampling

`--graphsage_model` - path to pre-trained GraphSAGE model

`--classification_model` - path to pre-trained partitioning model

`--lr` - learning rate

`--model` - path to the whole model (don't use `graphsage_model` and `classification_model` if you use this)

`--cuda` - whether to use CUDA device

`--learn_method` - possible options `gap` (unsupervised), `sup_edge` (supervised)

`--num_steps` - if you want to run the training for less than an epoch, you can specify simply the number of training steps

Additionally there's `experiments.conf` file, which contains the paths to the files used for training - training edges, features and labeled training edges.
#### run_stream_partitioning.py - GCNsplit partitioning
`--learn_method` - learn method used for training

`--dataset` - dataset name

`--num_classes` - number of partitions

`--max_load` - maximum normalized load that any partition can reach

`--model` - model to use for partitioning

`--inf_b_sz` - number of edges to partition in each batch (window)

`--num_processes` - number of parallel partitioning processes 

`--edge_file_path` - path to the file containing edges to partition

`--feats_file_path` - path to the file containing features

`--sorted_inference` - whether to use HighestAvailable heuristic

`--with_train_adj` - whether to use a training graph as a neighbourhood context for each window; path to the training 
edges is defined in `experiments.conf`


#### run_evaluate.py - evaluation
`--input_file` - path to the file to evaluate (`*RESULTS.csv`) 
`--num_classes` - number of partitions
`--chunk_size=100000` - number of edges to read from the disk at once

### Datasets
Twitch, Deezer datasets are provided with this repository. Due to their sizes Reddit, Papers100m and Bitcoin
datasets can be provided upon request.

#### Training configuration

|   Dataset  |   Training   | No. Partitions | alpha |       beta      | No. epochs | Batch Size |
|:----------:|:------------:|:--------------:|:-----:|:---------------:|:----------:|:----------:|
|   Reddit   | Unsupervised |        2       |   1   | 10<sup>-5</sup> |     200    |  All nodes |
|   Reddit   | Unsupervised |        3       |   1   | 10<sup>-5</sup> |     200    |  All nodes |
|   Reddit   | Unsupervised |        4       |   1   | 10<sup>-5</sup> |     300    |  All nodes |
|   Reddit   | Unsupervised |        5       |   1   | 10<sup>-5</sup> |     300    |  All nodes |
|   Reddit   | Unsupervised |        6       |   1   | 10<sup>-5</sup> |     500    |  All nodes |
|   Reddit   | Unsupervised |        7       |   1   | 10<sup>-5</sup> |     500    |  All nodes |
|   Reddit   | Unsupervised |       10       |   1   | 10<sup>-5</sup> |     700    |  All nodes |
|   Reddit   |  Supervised  |        2       |   1   |        10       |     10     |  100 edges |
|   Reddit   |  Supervised  |        3       |   1   |        0        |     15     |  100 edges |
|   Reddit   |  Supervised  |        4       |   1   |        0        |     20     |  100 edges |
|   Reddit   |  Supervised  |        5       |   1   |        0        |     30     |  100 edges |
|   Reddit   |  Supervised  |        6       |   1   |        0        |     35     |  100 edges |
|   Reddit   |  Supervised  |        7       |   1   |        0        |     100    |  100 edges |
|   Reddit   |  Supervised  |       10       |   1   |        0        |     150    |  100 edges |
|   Twitch   | Unsupervised |        6       |   1   | 10<sup>-5</sup> |      3     |     20     |
|   Twitch   |  Supervised  |        6       |   1   |        0        |     15     | 2000 edges |
|   Deezer   | Unsupervised |        6       |   1   | 10<sup>-6</sup> |      3     |     50     |
|   Deezer   |  Supervised  |        6       |   1   |        0        |     200    |  200 edges |
|   Bitcoin  | Unsupervised |        6       |   1   | 10<sup>-5</sup> |      2     |   5 nodes  |
| Papers100m | Unsupervised |        6       |   1   | 10<sup>-5</sup> |      1     |  15 nodes  |
