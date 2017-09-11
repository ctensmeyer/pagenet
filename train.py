#!/usr/bin/python

import os
import sys
import collections
import argparse
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import caffe
import cv2
import random
import scipy.ndimage as nd



def safe_mkdir(_dir):
	try:
		os.makedirs(_dir)
	except:
		pass


def dump_debug(out_dir, data, dump_images=False):
	pred_image_dir = os.path.join(out_dir, 'pred_images')
	safe_mkdir(pred_image_dir)

	for idx in xrange(len(data['images'])):
		fn = data['filenames'][idx]
		preds = data['predictions'][idx] 

		fn_base = fn.replace('/', '_')[:-4] 
		out_fn = os.path.join(pred_image_dir, fn_base + ".png")
		cv2.imwrite(out_fn, 255 * preds)


def predict(network, im, output_blob, args):
	if im.ndim > 2:
		im = np.transpose(im, axes=(2, 0, 1))
	network.blobs["data"].data[0,:,:,:] = im
	network.forward()

	response = network.blobs[output_blob].data[0,:].copy()
	return np.argmax(response, axis=0)


def iou(im1, im2):
	num_intersect = np.sum(np.logical_and(im1, im2))
	num_union = num_intersect + np.sum(np.logical_xor(im1, im2))
	return float(num_intersect) / num_union
	

def prf(im1, im2):
	num_intersect = np.sum(np.logical_and(im1, im2))
	num_1 = np.sum(im1)
	num_2 = np.sum(im2)
	p = num_intersect / float(num_1)
	r = num_intersect / float(num_2)
	f = (2 * p * r) / (p + r) if (p + r) else 0
	return p, r, f
	

def update_predictions(net, data, args):
	print "Starting Predictions"

	total_iou = 0
	total_p = 0
	total_r = 0
	total_f = 0
	for idx in xrange(len(data['images'])):
		im = cv2.resize(data['images'][idx], (args.image_size, args.image_size))

		outputs = predict(net, im, 'out', args)
		data['predictions'][idx] = outputs.copy()

		width, height = data['original_size'][idx]
		outputs = cv2.resize(outputs, (width, height), interpolation=cv2.INTER_NEAREST)
		total_iou += iou(outputs, data['original_gt'][idx])

		p, r, f = prf(outputs, data['original_gt'][idx])
		total_p += p
		total_r += r
		total_f += f


		if idx and idx % args.print_count == 0:
			print "\tPredicted %d/%d" % (idx, len(data['images']))
	avg_iou = total_iou / len(data['images'])
	avg_p = total_p / len(data['images'])
	avg_r = total_r / len(data['images'])
	avg_f = total_f / len(data['images'])
	return avg_iou, avg_p, avg_r, avg_f


def load_data(manifest, _dir, size, color=False):
	dataset = collections.defaultdict(list)
	file_list = map(lambda s: s.strip(), open(manifest, 'r').readlines())
	for line in file_list:
		tokens = line.split(',')
		f = tokens[0]
		coords = map(float, tokens[1:9])

		dataset['filenames'].append(f)

		resolved = os.path.join(_dir, f)
		im = cv2.imread(resolved, 1 if color else 0)
		gt = np.zeros(im.shape[:2], dtype=np.uint8)
		cv2.fillPoly(gt, np.array(coords).reshape((4, 2)).astype(np.int32)[np.newaxis,:,:], 1)
		if im is None:
			raise Exception("Error loading %s" % resolved)
		height, width = im.shape[:2]
		im = cv2.resize(im, (size, size))
		dataset['original_gt'].append(gt)
		gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
		dataset['images'].append(im)
		dataset['original_size'].append( (width, height) )  # opencv does (w,h)
		dataset['gt'].append(gt)

	return dataset


def preprocess_data(data, args):
	for idx in xrange(len(data['images'])):
		im = data['images'][idx]
		im = args.scale * (im - args.mean)
		data['images'][idx] = im

		gt = data['gt'][idx]
		data['predictions'].append(gt.copy())


def get_solver_params(f):
	max_iters = 0
	snapshot = 0

	for line in open(f).readlines():
		tokens = line.split()
		if tokens[0] == 'max_iter:':
			max_iters = int(tokens[1])
		if tokens[0] == 'snapshot:':
			snapshot = int(tokens[1])
	return max_iters, snapshot


def presolve(net, args):
	net.blobs["data"].reshape(args.batch_size, 3 if args.color else 1, args.image_size, args.image_size)
	net.blobs["gt"].reshape(args.batch_size, 1, args.image_size, args.image_size)


def set_input_data(net, data, args):
	for batch_idx in xrange(args.batch_size):
		im_idx = random.randint(0, len(data['images']) - 1)
		im = data['images'][im_idx]
		gt = data['gt'][im_idx]

		if im.ndim > 2:
			im = np.transpose(im, (2, 0, 1))

		net.blobs["data"].data[batch_idx,:,:,:] = im
		net.blobs["gt"].data[batch_idx,0,:,:] = gt


def main(args):
	
	train_data = load_data(args.train_manifest, args.dataset_dir, args.image_size, args.color)
	val_data = load_data(args.val_manifest, args.dataset_dir, args.image_size, args.color)

	preprocess_data(train_data, args)
	preprocess_data(val_data, args)

	print "Done loading data"

	solver = caffe.SGDSolver(args.solver_file)
	max_iters, snapshot_interval = get_solver_params(args.solver_file)

	presolve(solver.net, args)
	train_iou, val_iou = [], []
	train_p, val_p = [], []
	train_r, val_r = [], []
	train_f, val_f = [], []

	for iter_num in xrange(max_iters + 1):
		set_input_data(solver.net, train_data, args)
		solver.step(1)

		if iter_num and iter_num % snapshot_interval == 0:
			print "Validation Prediction: %d" % iter_num
			avg_iou, avg_p, avg_r, avg_f = update_predictions(solver.net, val_data, args)
			val_iou.append((iter_num, avg_iou))
			val_p.append((iter_num, avg_p))
			val_r.append((iter_num, avg_r))
			val_f.append((iter_num, avg_f))
			if args.debug_dir:
				print "Dumping images"
				out_dir = os.path.join(args.debug_dir, 'val_%d' % iter_num)
				dump_debug(out_dir, val_data)

		if iter_num >= args.min_interval and iter_num % args.gt_interval == 0:

			print "Train Prediction: %d" % iter_num
			avg_iou, avg_p, avg_r, avg_f = update_predictions(solver.net, train_data, args)
			train_iou.append((iter_num, avg_iou))
			train_p.append((iter_num, avg_p))
			train_r.append((iter_num, avg_r))
			train_f.append((iter_num, avg_f))
				
	print "Train IOU: ", train_iou
	print
	print "Val IOU: ", val_iou
	if args.debug_dir:
		plt.plot(*zip(*train_iou), label='train')
		plt.plot(*zip(*val_iou), label='val')
		plt.legend()
		plt.savefig(os.path.join(args.debug_dir, 'iou.png'))


		plt.clf()
		plt.plot(*zip(*train_p), label='train')
		plt.plot(*zip(*val_p), label='val')
		plt.legend()
		plt.savefig(os.path.join(args.debug_dir, 'precision.png'))

		plt.clf()
		plt.plot(*zip(*train_r), label='train')
		plt.plot(*zip(*val_r), label='val')
		plt.legend()
		plt.savefig(os.path.join(args.debug_dir, 'recall.png'))

		plt.clf()
		plt.plot(*zip(*train_f), label='train')
		plt.plot(*zip(*val_f), label='val')
		plt.legend()
		plt.savefig(os.path.join(args.debug_dir, 'fmeasure.png'))

		_ = update_predictions(solver.net, train_data, args)
		out_dir = os.path.join(args.debug_dir, 'train_final')
		dump_debug(out_dir, train_data, True)

		_ = update_predictions(solver.net, val_data, args)
		out_dir = os.path.join(args.debug_dir, 'val_final')
		dump_debug(out_dir, val_data, True)

		for name, vals in zip(['train_iou', 'val_iou', 'train_p', 'val_p', 
							   'train_r', 'val_r', 'train_f', 'val_f'], 
							  [train_iou, val_iou, train_p, val_p, 
							   train_r, val_r, train_f, val_f]):
			fd = open(os.path.join(args.debug_dir, "%s.txt" % name), 'w')
			fd.write('%r\n' % vals)
			fd.close()

	
def get_args():
	parser = argparse.ArgumentParser(description="Outputs binary predictions")

	parser.add_argument("solver_file", 
				help="The solver.prototxt")
	parser.add_argument("dataset_dir",
				help="The dataset to be evaluated")
	parser.add_argument("train_manifest",
				help="txt file listing images to train on")
	parser.add_argument("val_manifest",
				help="txt file listing images for validation")

	parser.add_argument("--gpu", type=int, default=0,
				help="GPU to use for running the network")

	parser.add_argument("-m", "--mean", type=float, default=127.,
				help="Mean value for data preprocessing")
	parser.add_argument("-s", "--scale", type=float, default=1.,
				help="Optional pixel scale factor")
	parser.add_argument("-b", "--batch-size", default=2, type=int, 
				help="Training batch size")
	parser.add_argument("-c", "--color",  default=False,  action='store_true', 
				help="Training batch size")

	parser.add_argument("--image-size", default=256, type=int, 
				help="Size of images for input to training/prediction")

	parser.add_argument("--gt-interval", default=5000, type=int, 
				help="Interval for Debug")
	parser.add_argument("--min-interval", default=5000, type=int, 
				help="Miniumum iteration for Debug")

	parser.add_argument("--debug-dir", default='debug', type=str, 
				help="Dump images for debugging")
	parser.add_argument("--print-count", default=10, type=int, 
				help="Dump images for debugging")

	args = parser.parse_args()
	print args

	return args
			

if __name__ == "__main__":
	args = get_args()

	if args.gpu >= 0:
		caffe.set_device(args.gpu)
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()

	main(args)


