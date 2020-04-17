# D3S - A Discriminative Single Shot Segmentation Tracker [CVPR2020]

Python (PyTorch) implementation of the D3S tracker, presented at CVPR 2020.

## Publication:
Alan Lukežič, Jiří Matas and Matej Kristan.
<b>D3S - A Discriminative Single Shot Segmentation Tracker.</b>
<i>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2020</i>.</br>
[Paper](https://arxiv.org/abs/1911.08862) </br>

<b>BibTex citation:</b></br>
@InProceedings{Lukezic_CVPR_2020,<br>
Title = {D3S - A Discriminative Single Shot Segmentation Tracker},<br>
Author = {Lukezic, Alan and Matas, Jiri and Kristan, Matej},<br>
Booktitle = {CVPR},<br>
Year = {2020}<br>
}

## Summary of the D3S tracker
Template-based discriminative trackers are currently the dominant tracking paradigm due to their robustness, but are restricted to bounding box tracking and a limited range of transformation models, which reduces their localization accuracy. We propose a discriminative single-shot segmentation tracker -- D3S, which narrows the gap between visual object tracking and video object segmentation. A single-shot network applies two target models with complementary geometric properties, one invariant to a broad range of transformations, including non-rigid deformations, the other assuming a rigid object to simultaneously achieve high robustness and online target segmentation. Without per-dataset finetuning and trained only for segmentation as the primary output, D3S outperforms all trackers on VOT2016, VOT2018 and GOT-10k benchmarks and performs close to the  state-of-the-art trackers on the TrackingNet. D3S outperforms the leading segmentation tracker SiamMask on video  object segmentation benchmarks and performs on par with top video object segmentation algorithms, while running an order of magnitude faster, close to real-time.

<p style="width:100%, text-align:center"><a href="url"><img src="https://raw.githubusercontent.com/alanlukezic/d3s/master/pytracking/utils/d3s-architecture.png" width="640"></a></p>

## Installation

#### Clone the GIT repository.  
```bash
git clone https://github.com/alanlukezic/d3s.git .
```

#### Install dependencies
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```pytracking```).  
```bash
bash install.sh conda_install_path pytracking
```
To install the dependencies on a Windows machine, use the `install.bat` script.
The pre-trained network for the D3S is not part of this repository. You can download it [here](http://data.vicos.si/alanl/d3s/SegmNet.pth.tar).

The tracker was tested on the Ubuntu 16.04 machine with a NVidia GTX 1080 graphics card and cudatoolkit version 9.
It was tested on Window 10 as well, but network training is tested on Linux only.

#### Test the tracker
1.) Specify the path to the D3S [pre-trained segmentation network](http://data.vicos.si/alanl/d3s/SegmNet.pth.tar) by setting the `params.segm_net_path` in the `pytracking/parameters/segm/default_params.py`. <br/>
2.) Specify the path to the VOT 2018 dataset by setting the `vot18_path` in the `pytracking/evaluation/local.py`. <br/>
3.) Activate the conda environment
```bash
conda activate pytracking
```
4.) Run the script pytracking/run_tracker.py to run D3S using VOT18 sequences.  
```bash
cd pytracking
python run_tracker.py segm default_params --dataset vot18 --sequence <seq_name> --debug 1
```

### Evaluate the tracker using VOT
We provide a VOT Matlab toolkit integration for the D3S tracker. There is the `tracker_D3S.m` Matlab file in the `pytracking/utils`, which can be connected with the toolkit. It uses the `vot_wrapper.py` script to integrate the tracker to the toolkit.

#### Training the network
The D3S is pre-trained for segmentation task only on the YouTube VOS dataset. Download the VOS training dataset (2018 version) and copy the files `vos-list-train.txt` and `vos-list-val.txt` from `ltr/data_specs` to the `train` directory of the VOS dataset. 
Set the `vos_dir` variable in `ltr/admin/local.py` to the VOS `train` directory on your machine. 
Download the bounding boxes from [this link](http://data.vicos.si/alanl/d3s/rectangles.zip) and copy them to the sequence directories.
Run training by running the following command:
```bash
python run_training.py segm segm_default
```

## Pytracking
This is a modified version of the python framework pytracking based on **PyTorch**. We would like to thank the authors Martin Danelljan and Goutam Bhat for providing such a great framework.

## Video
Check out the [video](https://www.youtube.com/watch?v=E3mN_hCRHu0) with tracking and segmentation results of the D3S tracker.

## Contact
* Alan Lukežič (email: alan.lukezic@fri.uni-lj.si)
