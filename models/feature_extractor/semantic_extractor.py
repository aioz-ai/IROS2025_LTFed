import torch
from ultralytics import YOLO
from models.feature_extractor.utils import crop_tensor_image, padding_tensor
import os

class Semantic_Feature_Extractor(torch.nn.Module):
	def __init__(self, feature_extractor_transform, max_boxes=5):
		super(Semantic_Feature_Extractor, self).__init__()
		self.object_detection = YOLO(os.path.join("pretrained_models", "yolov8n.pt"))
		self.feature_extractor_transform = feature_extractor_transform
		self.max_boxes = max_boxes
		self.object_detection.model.fuse(verbose=False)

	def forward(self, x, feature_extractor):
		detection_results = self.object_detection.predict(x, verbose=False, conf=0.001)
		batch_rois_list = []
		for index in range(len(detection_results)):
			detections = detection_results[index].boxes
			batch_rois = crop_tensor_image(x[index], detections)
			if batch_rois is not None:
				batch_rois = self.feature_extractor_transform(batch_rois)
			batch_rois_list.append(batch_rois)
		
		feature_list = []
		for batch_rois in batch_rois_list:
			if batch_rois is not None:
				if batch_rois.shape[0] > self.max_boxes:
					batch_rois = batch_rois[:self.max_boxes]
				features = feature_extractor(batch_rois)['out']
				features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).reshape(features.shape[0], -1)
				# features = torch.flatten(features, start_dim=1)
				features = padding_tensor(features, size=self.max_boxes)
			else:
				features = torch.zeros(self.max_boxes, 1280).type(x.type())
			feature_list.append(features)
		
		# global features
		global_feats = crop_tensor_image(x, None)
		global_feats = self.feature_extractor_transform(global_feats)
		global_feats = feature_extractor(global_feats)['out']
		global_feats = torch.nn.functional.adaptive_avg_pool2d(global_feats, (1, 1)).reshape(global_feats.shape[0], -1)
		global_feats = torch.unsqueeze(global_feats, 1)

		# concat local with global feats
		return torch.cat((torch.stack(feature_list), global_feats), 1), detection_results
	def train(self, mode=True):
		self.training = mode
		self.object_detection.model.train(mode)
		self.object_detection.predictor = None