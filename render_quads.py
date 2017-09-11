
import sys
import numpy as np
import cv2
import os

if len(sys.argv) < 4:
	print "python %s manifest.txt dataset_dir out_dir" % __file__
	exit()

manifest_file = sys.argv[1]
dataset_dir = sys.argv[2]
out_dir = sys.argv[3]

try:
	os.makedirs(out_dir)
except:
	pass

file_list = map(lambda s: s.strip(), open(manifest_file, 'r').readlines())
for line in file_list:
	tokens = line.split(',')
	f = tokens[0]
	coords = map(float, tokens[1:9])

	resolved = os.path.join(dataset_dir, f)
	im = cv2.imread(resolved, 0)
	gt = np.zeros(im.shape, dtype=np.uint8)
	cv2.fillPoly(gt, np.array(coords).reshape((4, 2)).astype(np.int32)[np.newaxis,:,:], 255)

	out_fn = os.path.join(out_dir, f.replace('/', '_'))[:-4] + ".png"
	cv2.imwrite(out_fn, gt)

