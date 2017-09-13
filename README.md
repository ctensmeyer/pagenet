# PageNet

PageNet is a Deep Learning system that takes in an image with a document in it and returns a quadrilateral representing the main page region.  We trained PageNet using the library [Caffe](caffe.berkeleyvision.org).  For details, see our [paper](https://arxiv.org/abs/1709.01618).

## Usage

There are three scripts in this repo.  One for training networks, one for predictions using pre-trained networks, and one for rendered quadrilateral regions.

### Testing Pretrained Models

We have provided two pretrained models from our paper.  One model is trained on the CBAD dataset and the other is trained on a private collection of Ohio Death Records provided by [Family Search](https://www.familysearch.org/).

`test_pretrained.py` has the following usage

```
usage: test_pretrained.py [-h] [--out-dir OUT_DIR] [--gpu GPU]
                          [--print-count PRINT_COUNT]
                          image_dir manifest model out_file

Outputs binary predictions

positional arguments:
  image_dir             The directory where images are stored
  manifest              txt file listing images relative to image_dir
  model                 [cbad|ohio]
  out_file              Output file

optional arguments:
  -h, --help            show this help message and exit
  --out-dir OUT_DIR
  --gpu GPU             GPU to use for running the network
  --print-count PRINT_COUNT
                        Print interval
```
`image_dir` is the directory containing images to predict.  The file paths listed in `manifest` are relative to `image_dir` and are listed one per line.  `model` should be either `cbad` or `ohio` to select which trained model to use.  `out_file` will list the coordinates of the quadrilaterals predicted by PageNet for each of the input images.

`--gpu` is for passing the device ID of the GPU to use.  If it is negative, CPU mode is used.  Specifying `--out-dir` will allow you to dump both the raw and post processed predictions as images.  


### Training Your Own Models

`train.py` has the following usage

```
usage: train.py [-h] [--gpu GPU] [-m MEAN] [-s SCALE] [-b BATCH_SIZE] [-c]
                [--image-size IMAGE_SIZE] [--gt-interval GT_INTERVAL]
                [--min-interval MIN_INTERVAL] [--debug-dir DEBUG_DIR]
                [--print-count PRINT_COUNT]
                solver_file dataset_dir train_manifest val_manifest

Outputs binary predictions

positional arguments:
  solver_file           The solver.prototxt
  dataset_dir           The dataset to be evaluated
  train_manifest        txt file listing images to train on
  val_manifest          txt file listing images for validation

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             GPU to use for running the network
  -m MEAN, --mean MEAN  Mean value for data preprocessing
  -s SCALE, --scale SCALE
                        Optional pixel scale factor
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Training batch size
  -c, --color           Training batch size
  --image-size IMAGE_SIZE
                        Size of images for input to training/prediction
  --gt-interval GT_INTERVAL
                        Interval for Debug
  --min-interval MIN_INTERVAL
                        Miniumum iteration for Debug
  --debug-dir DEBUG_DIR
                        Dump images for debugging
  --print-count PRINT_COUNT
                        How often to print progress
```
`solver_file` points to a caffe solver.prototxt file.  Such a file is included in the repo.  The training script expects that the network used for training to begin and end like the included `train_val.prototxt` file, but the middle layers can be changed.
`dataset_dir` is the directory containing the training and validation images.  The file paths listed in `train_manifest` and `val_manifest` are relative to `dataset_dir` and are listed one per line.

`--gpu` is for passing the device ID of the GPU to use.  If it is negative, CPU mode is used.   `--debug-dir` defaults to `debug` and if it is not the empty string, predictions and metrics will be dumped at intervals specified by `--gt-interval` and `--min-interval`.  This can help with selecting the best model from the snapshots.

The optional arguments have reasonable defaults.  If you're curious about their exact meaning, I suggest you look at the code.

### Testing Your Own Models

If you have trained your own model with `train.py`, you can test it with `test.py`.  The usage is
```
usage: test.py [-h] [--out-dir OUT_DIR] [--gpu GPU] [-c] [-m MEAN] [-s SCALE]
               [--image-size IMAGE_SIZE] [--print-count PRINT_COUNT]
               net_file weight_file dataset_dir test_manifest out_file

Outputs binary predictions

positional arguments:
  net_file              The deploy.prototxt
  weight_file           The .caffemodel
  dataset_dir           The dataset to be evaluated
  test_manifest         Images to predict
  out_file              output file listing quad regions

optional arguments:
  -h, --help            show this help message and exit
  --out-dir OUT_DIR     Dump images
  --gpu GPU             GPU to use for running the network
  -c, --color           Training batch size
  -m MEAN, --mean MEAN  Mean value for data preprocessing
  -s SCALE, --scale SCALE
                        Optional pixel scale factor
  --image-size IMAGE_SIZE
                        Size of images for input to prediction
  --print-count PRINT_COUNT
                        Print interval

```

The optional arguments for this script mirror those for `train.py` and should be set to the same values.  The required arguments are the same as for `test_pretrained.py`, except you manually specify `network file` (e.g., `train_val.prototxt`) and the `weight_file`.

### Rendering Masks

The usage for `render_quads.py` is
```
python render_quads.py manifest dataset_dir out_dir
```

`manifest` lists the image file path and quadrilateral coordinates.  It should be the `out_file` of `test_pretrained.py`.  The filepaths in `manifest` are relative to `dataset_dir`.  `out_dir` is an output directory where quadrilateral region images are written


## Dependencies

The python scripts depend on OpenCV 3.2, Matplotlib, Numpy, and Caffe.

## Docker

For those who don't want to install the dependencies, I have created a docker image to run this code. You must have the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin installed to use it though you can still run our models on CPU (not recommended).

The usage for the docker container is

```
nvidia-docker run -v $HOST_WORK_DIRECTORY:/data tensmeyerc/icdar2017:pagenet python $SCRIPT $ARGS
```

`$HOST_WORK_DIRECTORY` is a directory on your machine that is mounted on `/data` inside of the docker container (using -v).  It's the only way to expose files to the docker container.
`$SCRIPT` is one of the scripts described above.  `$ARGS` are the normal arguments you pass to the python script.  Note that any file paths passed as arguments must begin with `/data` to be visible to the docker container.
There is no need to download the container ahead of time.  If you have docker and nvidia-docker installed, running the above commands will pull the docker image (~2GB) if it has not been previously pulled.

## Baseline Methods

For those wanting to replicate the baselines we reported in our paper, we've included the two scripts we used. 

`baselines/grabCutCropEval.py` evaluates using the Grab Cut approach described in the paper. As input arguments, it takes the groudtruth file, image  directory, directory to store intermediate results, and the number of threads to run the script on. It produces a file which is the ground truth file name with `_fullgrab.res` appended.

```
python grabCutCropEval.py gtFile imageDir interDir numThreads
```

`baselines/noCutMeanCutCropEval.py` evaulates using full image (no cropping) and a mean quadrilateral. The mean quadrilateral is computed from the ground truth if not given in the arguments. It prints the computed mean so it can be reused. As input arguments, the script takes the ground truth file, image directory, and optionally the four (normalized 0.0 - 1.0) quadrilateral points (in clockwise order, starting at top left). It produces two files which are the ground truth file name with `_fullno.res` and `_fullmean.res` appended. 

```
python noCutMeanCutCropEval.py gtFile imageDir [mean_x1 mean_y1 mean_x2 mean_y2 mean_x3 mean_y3 mean_x4 mean_y4]
```


## Citation

If you find this code useful to your research, please cite our paper:

```
@article{tensmeyer2017_pagenet,
  title={PageNet: Page Boundary Extraction in Historical Handwritten Documents},
  author={Tensmeyer, Chris and Davis, Brian and Wigington, Curtis and Lee, Iain and Barrett, Bill},
  journal={arXiv preprint arXiv:1709.01618},
  year={2017},
}
```
