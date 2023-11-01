# The cross-modality survival prediction method of glioblastoma based on dual-graph neural networks

This repository is the work of "The cross-modality survival prediction method of glioblastoma based on dual-graph neural networks" based on **pytorch** implementation. 

Note that, the HisGAN model in `models` file will open when the paper is accept.

You could click the link to access the [paper](https://arxiv.org/). The multimodal FeTS dataset could be acquired from [here](https://github.com/FETS-AI/Challenge).


<div align="center">  

 <img src="https://github.com/JalexDooo/GNN_Bracls/tree/main/pairplot1.jpg"
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


### Data preprocessing

```
python3 -u main.py process_data
```

### Training

Multiply gpus training is recommended. The total training time take less than 2 hours in gtxforce 2080Ti. Training like this:

```
python3 -u main.py trainGraph
```

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

