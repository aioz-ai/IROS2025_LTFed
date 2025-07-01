import torch
from torchvision.transforms import functional as F
# Define function to crop ROIs from tensor image
def crop_tensor_image(tensor_image, bounding_boxes=None, target_size=(224, 224)):
	if bounding_boxes is None:
		return F.resize(tensor_image, target_size)
	cropped_rois = []
	for box in bounding_boxes:
		xmin, ymin, xmax, ymax = map(int, box.xyxy[0]) 
		
		# Crop ROI from tensor image
		cropped_roi = tensor_image[:, ymin:ymax, xmin:xmax]  # Crop along H, W dimensions
		cropped_roi = F.resize(cropped_roi, target_size)
		cropped_rois.append(cropped_roi)
	if len(cropped_rois) > 0:
		# Stack cropped ROIs into a batch tensor
		batch_rois = torch.stack(cropped_rois)
	else:
		batch_rois = None
	return batch_rois

def padding_tensor(features, size=10):
	if features.shape[0] < size:
		pad_size = size - features.shape[0]
		pad = torch.zeros(pad_size, features.shape[1]).type(features.type())
		features = torch.cat((features, pad), dim=0)
	else:
		features = features[:size]
	return features

# def get_bbox_point(bounding_boxes):
# 	results = []
# 	for box in bounding_boxes:
# 		xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
# 		results.append((xmin, ymin, xmax, ymax))
# 	return results

class NumpyToTensor:
	""" Convert a ``np.array`` to a torch.tensor of the same type.

	Methods:
		__call__(): Apply F.to_tensor to np.array and torch.as_tensor to target grayscale image
	"""
	def __call__(self, image):
		image = F.to_tensor(image)
		return image