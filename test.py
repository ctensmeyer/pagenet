#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import caffe
import cv2
import random



def safe_mkdir(_dir):
	try:
		os.makedirs(_dir)
	except:
		pass


def predict(network, im, output_blob, args):
	network.blobs["data"].data[0,:,:,:] = im
	network.forward()

	response = network.blobs[output_blob].data[0,:].copy()
	return np.argmax(response, axis=0)


def presolve(net, args):
	net.blobs["data"].reshape(1, 3 if args.color else 1, args.image_size, args.image_size)
	net.blobs["gt"].reshape(1, 1, args.image_size, args.image_size)


def main(args):
	net = caffe.Net(args.net_file, args.weight_file, caffe.TEST)
	presolve(net, args)

	file_list = map(lambda s: s.strip(), open(args.test_manifest, 'r').readlines())
	fd = open(args.out_file, 'w')
	for idx, line in enumerate(file_list):
		if idx % args.print_count == 0:
			print "Processed %d/%d Images" % (idx, len(file_list))
		tokens = line.split(',')
		f = tokens[0]
		resolved = os.path.join(args.dataset_dir, f)
		im = cv2.imread(resolved, 1 if args.color else 0)

		_input = args.scale * (cv2.resize(im, (args.image_size, args.image_size)) - args.mean)
		if _input.ndim > 2:
			_input = np.transpose(_input, (2, 0, 1))
		raw = (255 * predict(net, _input, 'out', args)).astype(np.uint8)

		if args.out_dir:
			out_fn = os.path.join(args.out_dir, f.replace('/','_')[:-4] + "_raw.png")
			cv2.imwrite(out_fn, raw)

		post, coords = post_process(raw)
		for idx2 in [1, 2, 3, 0]:
			fd.write('%d,%d,' % (width * coords[idx2][0] / 256., height * coords[idx2][1] / 256.))
		fd.write('\n')

		if args.out_dir:
			out_fn = os.path.join(args.out_dir, f.replace('/','_')[:-4] + "_post.png")
			cv2.imwrite(out_fn, post)
	
def get_args():
	parser = argparse.ArgumentParser(description="Outputs binary predictions")

	parser.add_argument("net_file", 
				help="The deploy.prototxt")
	parser.add_argument("weight_file", 
				help="The .caffemodel")
	parser.add_argument("dataset_dir",
				help="The dataset to be evaluated")
	parser.add_argument("test_manifest",
				help="Images to predict")
	parser.add_argument("out_file",
				help="output file listing quad regions")
	parser.add_argument("--out-dir", default='', type=str, 
				help="Dump images")

	parser.add_argument("--gpu", type=int, default=0,
				help="GPU to use for running the network")
	parser.add_argument("-c", "--color",  default=False, action='store_true', 
				help="Training batch size")

	parser.add_argument("-m", "--mean", type=float, default=127.,
				help="Mean value for data preprocessing")
	parser.add_argument("-s", "--scale", type=float, default=0.0039,
				help="Optional pixel scale factor")
	parser.add_argument("--image-size", default=256, type=int, 
				help="Size of images for input to prediction")

	parser.add_argument("--print-count", default=10, type=int, 
				help="Print interval")

	args = parser.parse_args()
	print args

	return args
			

if __name__ == "__main__":
	args = get_args()
	safe_mkdir(args.out_dir)

	if args.gpu >= 0:
		caffe.set_device(args.gpu)
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()

	main(args)


