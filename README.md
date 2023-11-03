# The cross-modality survival prediction method of glioblastoma based on dual-graph neural networks

This repository is the work of "The cross-modality survival prediction method of glioblastoma based on dual-graph neural networks" based on **pytorch** implementation. 

Note that, the GNN model in `models` file open as **GraphNet**.

You could click the link to access the [paper](https://arxiv.org/). The multimodal BraTS-OS dataset could be acquired from [here](https://ipp.cbica.upenn.edu).


<div align="center">  

 <img src="https://github.com/JalexDooo/GNN_Bracls/blob/main/pairplot1.jpg"
     align=center/>
</div>

<center>The data.</center>


## Requirements
- python 3.6
- pytorch 1.8.1 or later CUDA version (ARGAN model requires 1.8.1 or later)
- torchvision
- nibabel
- SimpleITK
- matplotlib
- fire
- Pillow

### Data
Dataset acquisition from [BraTS20 Challenge](https://www.med.upenn.edu/cbica/brats2020/). We comply with the CBICA dataset usage standards, and data acquisition requires researchers to apply on the website.

And the online validation evaluation of the results is at the address - (https://ipp.cbica.upenn.edu)


### Data preprocessing
This step is to get features of nodes and edges in G1 and G2.
If `./preprocess/` file is not exist, please create it.

G1 as node_features and edge_index. G2 as node_features2 and edge_index2.

```python
python3 -u main.py process_data
```

The features are saved AS

node_features : `'./preprocess/'+{sample}+'_node_features_{survival}_.npy'`

edge_index : `'./preprocess/'+{sample}+'_edge_index_{survival}_.npy'`

node_features2 : `'./preprocess/'+{sample}+'_node_features2_{survival}_.npy'`

edge_index2 : `'./preprocess/'+{sample}+'_edge_index2_{survival}_.npy'`


### Data Standard
Function `model_intput_shape` in `main.py` shows the input data shape.

```python
def model_intput_shape():
    # bz is Batch Size
    node1 = torch.randn((bz, 6, 64)) # G1 node from node_features
    edge1 = torch.randn((bz, 2, 22)) # G1 node from edge_index

    node2 = torch.randn((bz, 3, 64)) # G2 node from node_features2
    edge2 = torch.randn((bz, 2, 9)) # G2 edge from edge_index2

    img = torch.randn((bz, 1, 128, 128, 64)) # MRI image
    seg = torch.randn((bz, 1, 128, 128, 64)) # MRI tumor segmentation

    model = models.GraphNet(1)

    age = None # age is encoding in node1 and edge1
    out = model(img, seg, age, node1, edge1, node2, edge2)
    print(out.shape) # [bz, 1]
```

### Training

Multiply gpus training is recommended. The total training time take less than 2 hours in gtxforce 2080Ti. Training like this:

```
python3 -u main.py trainGraph
```

The path included in the function needs to be modified.

### Test (CPU version)

You could obtain the resutls as paper reported by running the following code:

```
python3 main.py train_val
```
Then make a submission to the online evaluation server.

## Citation

If you use our code or model in your work or find it is helpful, please cite the paper:
```
***Unknown***
```

