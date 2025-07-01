import torch
import torch.nn as nn
import torchvision.models as models
import tqdm

from utils.optim import get_optimizer, get_lr_scheduler
from models.utils import freeze_model
from ..model import Model
from ..feature_extractor.semantic_extractor import Semantic_Feature_Extractor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision import transforms
from models.attention.tri_attention import TriAttention
from models.attention.tc import TCNet

NUMBER_CLASSES = 1
# def round_tensor(x, decimals):
#     return torch.round(x * 10**decimals) / (10**decimals)

class LTFed_Net(nn.Module):
    def __init__(self):
        super(LTFed_Net, self).__init__()
        self.feature_extractor = IntermediateLayerGetter(models.mobilenet_v2(pretrained=True), return_layers={'features': 'out'})
        self.feature_extractor_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225]),
									])
        self.semantic_extractor = Semantic_Feature_Extractor(self.feature_extractor_transform)
        self.attention = TriAttention(1280, 1280, 1, 512, 1, 32, 2, 1)

        self.gamma = 2
        t_net = []

        for i in range(self.gamma):
            t_net.append(TCNet(1280, 1280, 1, 512, 512, 2, 1, dropout=[.2, .5], k=2))

        self.t_net = nn.ModuleList(t_net)

        self.fc = nn.Linear(1024, NUMBER_CLASSES)

        freeze_model(self.feature_extractor)
        freeze_model(self.semantic_extractor.object_detection.model)

    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def forward(self, x, previous_x, previous_y):
        semantic_feature, detection_results = self.semantic_extractor(x, self.feature_extractor)
            
        previous_x_size =list(previous_x.shape[:2])
        previous_x_image_size = list(previous_x.shape[2:])
        previous_x = previous_x.reshape([-1] + previous_x_image_size)
        previous_feature = self.feature_extractor(previous_x)['out']
        previous_feature = torch.nn.functional.adaptive_avg_pool2d(previous_feature, (1, 1)).reshape(previous_feature.shape[0], -1)

        zero_indices = torch.nonzero(torch.sum(previous_x, dim=[1, 2, 3]) == 0).squeeze()
        previous_feature[zero_indices] = torch.zeros(1280).type(previous_feature.type())

        previous_feature = previous_feature.reshape(previous_x_size + [1280])

        att, _ = self.attention(semantic_feature, previous_feature, previous_y)

        att = torch.nan_to_num(att, nan=0.0)
        b_emb = [0] * self.gamma
        for g in range(self.gamma):
            b_emb[g] = self.t_net[g].forward_with_weights(semantic_feature, previous_feature, previous_y, att[:, :, :, :, g])

        attention_feature = torch.stack(b_emb, dim=1).sum(1)

        return self.fc(attention_feature), detection_results

class Driving_LTFed(Model):
    def __init__(self, criterion, metric, device,
                 optimizer_name="adam", lr_scheduler="sqrt", initial_lr=1e-3, epoch_size=1):
        super(Driving_LTFed, self).__init__()
        self.net = LTFed_Net()
        self.net.to(device)
        self.criterion = criterion[0]
        self.criterion_supp = criterion[1]
        self.metric = metric
        self.device = device

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

    def fit_batch(self, iterator, round=0, update=True):
        self.net.train()

        self.optimizer.zero_grad()
        x, y, previous_x, previous_y = next(iter(iterator))
        x = x.to(self.device)
        y = y.unsqueeze(-1).to(self.device)
        previous_x = previous_x.to(self.device)
        previous_y = previous_y.unsqueeze(2).type(previous_x.type())
        # y = round_tensor(y, decimals=4).type(x.type())
        y = y.type(x.type())
        predictions, _ = self.net(x, previous_x, previous_y)
        # predictions = round_tensor(predictions, decimals=4)
        loss = self.criterion(predictions, y)
        acc = self.metric[0](y.cpu().detach().numpy(), predictions.cpu().detach().numpy(), squared = False) # squared=False for RMSE, squared=True for MSE

        loss.backward()

        if update:
            self.optimizer.step()
            self.lr_scheduler.step()

        batch_loss = loss.item()
        batch_acc = acc.item()

        return batch_loss, batch_acc

    def evaluate_iterator(self, iterator, test_method="use_ground-truth"):
        epoch_loss = 0
        epoch_rmse = 0
        epoch_mae = 0

        self.net.eval()
        with torch.no_grad():
            for i, (x, y, previous_x, previous_y) in enumerate(tqdm.tqdm(iterator)):
                x = x.to(self.device)
                y = y.unsqueeze(-1).to(self.device)
                previous_x = previous_x.to(self.device)
                if test_method == "use_ground-truth":
                    previous_y = previous_y.unsqueeze(2).type(previous_x.type())
                elif test_method == "use_prediction":
                    # if the previous_y is full of zeros, a new image sequence, reinit previous_y with full of zeros.
                    if torch.all(previous_y == 0.0):
                        previous_y_list = previous_y.squeeze(0).tolist()
                    previous_y = torch.Tensor(previous_y_list).unsqueeze(0).unsqueeze(2).type(previous_x.type())
                # y = round_tensor(y, decimals=4).type(x.type())
                y = y.type(x.type())
                predictions, _ = self.net(x, previous_x, previous_y)
                # predictions = round_tensor(predictions, decimals=4)
                loss = self.criterion(predictions, y)

                rmse = self.metric[0](y.cpu().detach().numpy(), predictions.cpu().detach().numpy(), squared = False) # squared=False for RMSE, squared=True for MSE
                mae = self.metric[1](y.cpu().detach().numpy(), predictions.cpu().detach().numpy())

                epoch_loss += loss.item()
                epoch_rmse += rmse.item()
                epoch_mae += mae.item()

                if test_method == "use_prediction":
                    previous_y_list.pop(0)
                    previous_y_list.append(predictions.cpu().item())

        return epoch_loss / len(iterator), epoch_rmse / len(iterator), epoch_mae / len(iterator)
    
    def fit_iterator_one_epoch(self, iterator):
        epoch_loss = 0
        epoch_rmse = 0
        epoch_mae = 0

        self.net.train()

        for i, (x, y, previous_x, previous_y) in enumerate(tqdm.tqdm(iterator)):
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.unsqueeze(-1).to(self.device)
            previous_x = previous_x.to(self.device)
            previous_y = previous_y.unsqueeze(2).type(previous_x.type())
            # y = round_tensor(y, decimals=4).type(x.type())
            y = y.type(x.type())
            predictions, _ = self.net(x, previous_x, previous_y)
            # predictions = round_tensor(predictions, decimals=4)

            loss = self.criterion(predictions, y)

            rmse = self.metric[0](y.cpu().detach().numpy(), predictions.cpu().detach().numpy(), squared = False) # squared=False for RMSE, squared=True for MSE
            mae = self.metric[1](y.cpu().detach().numpy(), predictions.cpu().detach().numpy())

            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            epoch_loss += loss.item()
            epoch_rmse += rmse.item()
            epoch_mae += mae.item()

        return epoch_loss / len(iterator), epoch_rmse / len(iterator), epoch_mae / len(iterator)

    def inference(self, iterator, test_method="use_prediction"):
        self.net.eval()
        previous_y_list = [0] * iterator.dataset.num_previous_frames
        results = []
        detection_results_list = []
        with torch.no_grad():
            for i, (x, y, previous_x, previous_y) in enumerate(tqdm.tqdm(iterator)):
                x = x.to(self.device)
                y = y.unsqueeze(-1).to(self.device)
                previous_x = previous_x.to(self.device)
                previous_y = torch.Tensor(previous_y_list).unsqueeze(0).unsqueeze(2).type(previous_x.type())
                # y = round_tensor(y, decimals=4).type(x.type())
                y = y.type(x.type())
                predictions, detection_results = self.net(x, previous_x, previous_y)
                detection_results_list.append(detection_results[0])
                if test_method == "use_prediction":
                    previous_y_list.pop(0)
                    previous_y_list.append(predictions.cpu().item())
                results.append(predictions.cpu().item())
        return results, detection_results_list