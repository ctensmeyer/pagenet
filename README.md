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


### Training

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

`--gpu` is for passing the device ID of the GPU to use.  If it is negative, CPU mode is used.

The optional arguments have reasonable defaults.  If you're curious about their exact meaning, I suggest you look at the code.

### Rendering Masks

The usage for `render_quads.py` is
```
python render_quads.py manifest dataset_dir out_dir
```

`manifest` lists the image file path and quadrilateral coordinates.  It should be the `out_file` of `test_pretrained.py`.  The filepaths in `manifest` are relative to `dataset_dir`.  `out_dir` is an output directory where quadrilateral region images are written


## Dependencies

The python scripts depend on OpenCV 3.2, Matplotlib, 
