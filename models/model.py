from abc import ABC, abstractmethod
import torch
import time
import os
from models.utils import load_model, load_lr_scheduler, load_optimizer, save_model, save_lr_scheduler, save_optimizer

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit_iterator_one_epoch(self, iterator):
        pass

    @abstractmethod
    def fit_batch(self, iterator):
        pass

    @abstractmethod
    def evaluate_iterator(self, iterator):
        pass

    def update_from_model(self, model):
        """
        update parameters using gradients from another model
        :param model: Model() object, gradients should be precomputed;
        """
        for param_idx, param in enumerate(self.net.parameters()):
            param.grad = list(model.net.parameters())[param_idx].grad.data.clone()

        self.optimizer.step()
        self.lr_scheduler.step()

    def fit_batches(self, iterator, n_steps, round):
        global_loss = 0
        global_acc = 0

        for step in range(n_steps):
            batch_loss, batch_acc = self.fit_batch(iterator, round=round)
            global_loss += batch_loss
            global_acc += batch_acc

        return global_loss / n_steps, global_acc / n_steps

    def fit_iterator(self, train_iterator, val_iterator=None, n_epochs=1, path=None, verbose=0, logger_write_param=None):
        best_valid_acc = float('inf')

        for epoch in range(n_epochs):

            start_time = time.time()

            train_loss, train_rmse, train_mae = self.fit_iterator_one_epoch(train_iterator)
            if val_iterator:
                valid_loss, valid_rmse, valid_mae = self.evaluate_iterator(val_iterator)
                self.save(epoch, path, valid_rmse)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if val_iterator:
                if valid_rmse < best_valid_acc:
                    best_valid_acc = valid_rmse
                    if path:
                        self.save('best', path, best_valid_acc)

            if verbose:
                logger_write_param.write(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                logger_write_param.write(f'\tTrain Loss: {train_loss:.5f} | Train RMSE: {train_rmse:.5f} | Train MAE: {train_mae:.5f}')
                if val_iterator:
                     logger_write_param.write(f'\t Val. Loss: {valid_loss:.5f} |  Val. RMSE: {valid_rmse:.5f} | Val. MAE: {valid_mae:.5f}') 

    def get_param_tensor(self):
        param_list = []

        for param in self.net.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)
    
    def save(self, epoch, path, metric):
        model_path = os.path.join(path, "epoch_{}.pth".format(epoch))
        save_model(model=self.net, round_idx=epoch, metric=metric, model_path=model_path)
        model_path = os.path.join(path, "epoch_{}_optimizer.pth".format(epoch))
        save_optimizer(optimizer=self.optimizer, round_idx=epoch, model_path=model_path)
        model_path = os.path.join(path, "epoch_{}_lr_scheduler.pth".format(epoch))
        save_lr_scheduler(lr_scheduler=self.lr_scheduler, round_idx=epoch, model_path=model_path)

    def load(self, epoch, path):
        model_path = os.path.join(path, "epoch_{}.pth".format(epoch))
        load_model(model=self.net, model_path=model_path, device=self.device)
        model_path = os.path.join(path, "epoch_{}_optimizer.pth".format(epoch))
        load_optimizer(optimizer=self.optimizer, model_path=model_path, device=self.device)
        model_path = os.path.join(path, "epoch_{}_lr_scheduler.pth".format(epoch))
        load_lr_scheduler(lr_scheduler=self.lr_scheduler, model_path=model_path, device=self.device)