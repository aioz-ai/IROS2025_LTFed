from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
from pathlib import Path
import torchvision.transforms as transforms
from models.feature_extractor.utils import NumpyToTensor
import torch
from torchvision.transforms import Normalize
def random_flip(image, steering_angle):
	"""
	Randomly flipt the image left <-> right, and adjust the steering angle.
	"""
	if np.random.rand() < 0.5:
		image = cv2.flip(image, 1)
		steering_angle = -steering_angle
	return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
	"""
	Randomly shift the image virtially and horizontally (translation).
	"""
	trans_x = range_x * (np.random.rand() - 0.5)
	trans_y = range_y * (np.random.rand() - 0.5)
	steering_angle += trans_x * 0.002
	trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
	height, width = image.shape[:2]
	image = cv2.warpAffine(image, trans_m, (width, height))
	return image, steering_angle

def augment(image, steering_angle, range_x=100, range_y=10):
	"""
	Generate an augumented image and adjust steering angle.
	(The steering angle is associated with the center image)
	"""
	image, steering_angle = random_flip(image, steering_angle)
	image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
	return image, steering_angle

transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 127.5) - 1.0)])

class ComplexDrivingData(Dataset):
	def __init__(
			self, file_path, device,
			target_size=(320, 240), # (width, height)
			crop_size = (200, 200),
			color_mode='grayscale',
			data_root='data'):

		self.data_root = data_root
		self.data_file = Path(file_path)
		self.device = device
		self.target_size = target_size
		self.crop_size = crop_size

		# Idea = associate each filename with a corresponding steering or label
		self.filenames = []
		self.imgs = None
		self.ground_truths = []

		if self.data_file.is_file():
			self._load_data()

		if color_mode not in {'rgb', 'grayscale'}:
			raise ValueError('Invalid color mode:', color_mode,
							 '; expected "rgb" or "grayscale".')
		self.color_mode = color_mode

	def _load_data(self):
		if self.imgs is None:
			print("Loading data from {}...".format(self.data_file))
			with np.load(self.data_file) as data:
				filenames = data['file_names']
				ground_truths = data['ground_truths']
				sorted_indices = np.argsort(filenames)
				self.filenames = filenames[sorted_indices]
				self.ground_truths = ground_truths[sorted_indices]
			print("Filenames: ", self.filenames.shape)
			print("Ground truths: ", self.ground_truths.shape)
			print("Done!")

	def __len__(self):
		return self.ground_truths.shape[0]

	def __getitem__(self, index):
		img_filename = os.path.join(self.data_root, self.filenames[index])
		img = load_img(img_filename, self.color_mode == "grayscale", self.target_size, self.crop_size)
		steering_angle = self.ground_truths[index]
		return img, steering_angle

class TemporalComplexDrivingData(ComplexDrivingData):
	def __init__(
		self, file_path, device,
		target_size=(320, 240), # (width, height)
		crop_size = (200, 200),
		color_mode='grayscale',
		data_root='data',
		target_size_previous=(224, 224),
		train=True,
		num_previous_frames=10,
		inference_mode=False):
		super(TemporalComplexDrivingData, self).__init__(file_path, device, target_size, crop_size, color_mode, data_root)

		self.semantic_transform = transforms.Compose([
			NumpyToTensor()
			])
		self.feature_extractor_transform = transforms.Compose([Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225]),
									])
		self.target_size_previous = target_size_previous
		self.num_video_frames = 100 if not inference_mode else len(self.filenames)

		# in testing, stride = 1
		self.stride = 5 if train else 1
		self.num_previous_frames = num_previous_frames
		self.samples = []
		self._process_data()

	def _process_data(self):
		'''
		+ 1. Split images of each silos into short videos.
		+ 2. Process all shorts to pick input. Each short video:
			+ 2.1 stride to pick frames
			+ 2.2 each selected frames
				+ get previous frames with padding None
				+ get sterring_angles of these frames with padding None
		'''
		# NOTE: split data without shuffle (The first 70% is for training and the remaining 30% is for testing) to avoid interrupted image sequences
		# Finallize all file names of a long video
		filenames_fi = []
		for file_name in self.filenames:
			filenames_fi.append(os.path.join(self.data_root, file_name))
		self.filenames = np.array(filenames_fi)
		# Step 1: Split images of each silos into short videos.
		short_video_list = [self.filenames[i:i + self.num_video_frames] for i in range(0, len(self.filenames), self.num_video_frames)]
		short_video_ground_truth_list = [self.ground_truths[i:i + self.num_video_frames] for i in range(0, len(self.ground_truths), self.num_video_frames)]
		# Step 2: Process all shorts to pick input
		for short_video, short_video_ground_truth in zip(short_video_list, short_video_ground_truth_list):
			short_video = np.concatenate((np.array([None] * self.num_previous_frames), short_video))
			short_video_ground_truth = np.concatenate((np.array([0] * self.num_previous_frames), short_video_ground_truth))
			for selected_frame_index in range(self.num_previous_frames, len(short_video), self.stride):
				selected_frame = short_video[selected_frame_index]
				previous_frames = short_video[(selected_frame_index-self.num_previous_frames):selected_frame_index]
				steering_angle = short_video_ground_truth[selected_frame_index]
				previous_frames_steering_angle = short_video_ground_truth[(selected_frame_index-self.num_previous_frames):selected_frame_index]
				sample = {
					"file_path": selected_frame,
					"steering_angle": steering_angle,
					"previous_frame_path_list": previous_frames,
					"previous_frame_steering_angle": previous_frames_steering_angle
				}
				self.samples.append(sample)

	def __getitem__(self, index):
		sample = self.samples[index]
		img = load_img(sample['file_path'], self.color_mode == "grayscale", self.target_size, self.crop_size)
		img = self.semantic_transform(img)
		steering_angle = sample["steering_angle"]
		previous_frame_path_list = sample["previous_frame_path_list"]
		previous_frame_img_list = []
		for previous_frame_path in previous_frame_path_list:
			if previous_frame_path is not None:
				previous_frame_img = load_img(previous_frame_path, self.color_mode == "grayscale", self.target_size_previous, self.crop_size)
				previous_frame_img = self.semantic_transform(previous_frame_img)
				previous_frame_img = self.feature_extractor_transform(previous_frame_img)
			else:
				previous_frame_img = torch.zeros((1 if self.color_mode == "grayscale" else 3, self.target_size_previous[1], self.target_size_previous[0]))
			previous_frame_img_list.append(previous_frame_img)
		previous_steering_angle_list = sample["previous_frame_steering_angle"]
		return img, steering_angle, torch.stack(previous_frame_img_list), previous_steering_angle_list

	def __len__(self):
		return len(self.samples)

def load_img(path, grayscale=False, target_size=None, crop_size=None):
	"""
	Load an image.
	# Arguments
		path: Path to image file.
		grayscale: Boolean, whether to load the image as grayscale.
		target_size: Either `None` (default to original size)
			or tuple of ints `(img_width, img_height)`.
		crop_size: Either `None` (default to original size)
			or tuple of ints `(img_width, img_height)`.
	# Returns
		Image as numpy array.
	"""

	img = cv2.imread(path) # H W C
	if grayscale:
		if len(img.shape) != 2:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	if target_size:
		# target_size is WxH
		if (img.shape[1], img.shape[0]) != target_size:
			img = cv2.resize(img, target_size) # resize use W H

	if crop_size:
		img = central_image_crop(img, crop_size[0], crop_size[1])

	if grayscale:
		img = img.reshape((1, img.shape[0], img.shape[1]))
		img = np.asarray(img, dtype=np.float32)
	return img

def central_image_crop(img, crop_width=150, crop_heigth=150):
	"""
	Crop the input image centered in width and starting from the bottom
	in height.
	# Arguments:
		crop_width: Width of the crop.
		crop_heigth: Height of the crop.
	# Returns:
		Cropped image.
	"""
	half_the_width = int(img.shape[1] / 2)
	img = img[img.shape[0] - crop_heigth: img.shape[0],
			  half_the_width - int(crop_width / 2):
			  half_the_width + int(crop_width / 2)]
	return img

def get_iterator_complex_driving(file_path, device, batch_size=1, num_workers=0, temporal=False, num_previous_frames=10):
	"""
	returns an iterator over GAZEBO AND CRALA dataset batches
	:param file_path: path to .npz file containing a list of tuples
		 each of them representing a path to an image and it class
	:param device:
	:param batch_size:
	:param num_workers:
	:return: torch.utils.DataLoader object constructed from UDACITY-DRIVING dataset object
	"""
	train_mode = not ('test' in file_path or 'validation' in file_path)
	if temporal:
		dataset = TemporalComplexDrivingData(file_path, device=device, target_size=(640, 480), crop_size=None, color_mode='rgb', train=train_mode, num_previous_frames=num_previous_frames)
	else:
		raise NotImplementedError
	if train_mode:
		iterator = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
	else:
		iterator = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

	return iterator