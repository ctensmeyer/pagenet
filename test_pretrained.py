#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import caffe
import cv2
from process_pixel_labels import post_process

NET_FILE = './models/cbad_train_val.prototxt'
WEIGHT_FILE = './models/cbad_weights.caffemodel'


def safe_mkdir(_dir):
	try:
		os.makedirs(_dir)
	except:
		pass


def predict(network, im, output_blob, args):
	network.blobs["data"].data[0,:,:,:] = im
	network.forward()

	if args.model == 'ohio':
		# sigmoid
		response = network.blobs[output_blob].data[0,0,:,:].copy()
		response[response >= 0.5] = 1
		response[response <= 0.5] = 0
		return response
	else:
		# softmax
		response = network.blobs[output_blob].data[0,:].copy()
		return np.argmax(response, axis=0)



def presolve(net, args):
	net.blobs["data"].reshape(1, 3, 256, 256)
	net.blobs["gt"].reshape(1, 1, 256, 256)


def main(args):
	net = caffe.Net(NET_FILE, WEIGHT_FILE, caffe.TEST)
	presolve(net, args)

	file_list = map(lambda s: s.strip(), open(args.manifest, 'r').readlines())
	fd = open(args.out_file, 'w')
	for idx, line in enumerate(file_list):
		if idx % args.print_count == 0:
			print "Processed %d/%d Images" % (idx, len(file_list))
		tokens = line.split(',')
		f = tokens[0]
		resolved = os.path.join(args.image_dir, f)
		im = cv2.imread(resolved, 1)

		_input = 0.0039 * (cv2.resize(im, (256, 256)) - 127.)
		_input = np.transpose(_input, (2, 0, 1))
		raw = (255 * predict(net, _input, 'out', args)).astype(np.uint8)

		out_fn = os.path.join(args.out_dir, f.replace('/','_')[:-4] + "_raw.png")
		cv2.imwrite(out_fn, raw)

		post, coords = post_process(raw)
		print coords

		out_fn = os.path.join(args.out_dir, f.replace('/','_')[:-4] + "_post.png")
		cv2.imwrite(out_fn, post)

	
def get_args():
	parser = argparse.ArgumentParser(description="Outputs binary predictions")

	parser.add_argument("image_dir",
				help="The directory where images are stored")
	parser.add_argument("manifest",
				help="txt file listing images relative to image_dir")
	parser.add_argument("model",
				help="[cbad|ohio]")
	parser.add_argument("out_file", type=str, 
				help="Output file")

	parser.add_argument("--out-dir", type=str, default='out',
				help="")
	parser.add_argument("--gpu", type=int, default=0,
				help="GPU to use for running the network")
	parser.add_argument("--print-count", default=10, type=int, 
				help="Print interval")

	args = parser.parse_args()
	print args

	return args
			

if __name__ == "__main__":
	args = get_args()
	safe_mkdir(args.out_dir)

	if args.model == 'ohio':
		NET_FILE = './models/ohio_train_val.prototxt'
		WEIGHT_FILE = './models/ohio_weights.caffemodel'

	if args.gpu >= 0:
		caffe.set_device(args.gpu)
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()

	main(args)


