import torch
import os
def freeze_model(model):
    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False

def make_dir(path):
	""" Create a parent directory tree of a path

	Parameters:
		path (string, required): the input path

	Returns:
		None
	"""
	dir = os.path.dirname(path)
	if dir:
		os.makedirs(dir, exist_ok=True) 

def save_model(model, round_idx, metric, model_path):
    """ Save the current model weights.
    It includes: round, model_state and best_metric

    Note that we must save model weights to a temporal path + "_temp" first 
    to make sure the saving process is complete before renaming the target model path.

    Parameters:
        model_path (string, required): the path to save
    """
    make_dir(model_path)
    model_dict = {
        'round': round_idx,
        'model_state': model.state_dict(),
        'metric': metric
    }
    # save checkpoint
    torch.save(model_dict, model_path)

def load_model(model, model_path, device):
		""" Load an model weights.
		It includes: round, model_state and best_metric. Only loading if it exists.

		Parameters:
			model_path (string, required): the path to load
		"""
		model_data = torch.load(model_path, map_location=device)
		if ('model_state' in model_data):
			model.load_state_dict(model_data.get('model_state', model_data))
		if ('round' in model_data and 'metric' in model_data):
			return model_data['round'], model_data['metric']

def load_optimizer(optimizer, model_path, device):
    """ Load an optimizer parameters.
    It includes: round, optimizer_state. Only loading if it exists.

    Parameters:
        model_path (string, required): the path to load

    Return
        int
    """
    model_data = torch.load(model_path, map_location=device)
    if ('optimizer_state' in model_data):
        optimizer.load_state_dict(
            model_data.get('optimizer_state', model_data))
    if ('round' in model_data):
        return model_data['round']

def save_optimizer(optimizer, round_idx, model_path):
    """ Save the current model weights.
    It includes: round and optimizer_state

    Note that we must save the optimizer to a temporal path + "_temp" first 
    to make sure the saving process is complete before renaming the target model path.

    Parameters:
        model_path (string, required): the path to save
    """
    make_dir(model_path)
    model_dict = {
        'round': round_idx,
        'optimizer_state': optimizer.state_dict(),
    }
    # save checkpoint
    torch.save(model_dict, model_path)

def load_lr_scheduler(lr_scheduler, model_path, device):
    """ Load an lr_scheduler parameters.
    It includes: round, lr_scheduler_state. Only loading if it exists.

    Parameters:
        model_path (string, required): the path to load

    Return
        int
    """
    model_data = torch.load(model_path, map_location=device)
    if ('lr_scheduler_state' in model_data):
        lr_scheduler.load_state_dict(
            model_data.get('lr_scheduler_state', model_data))
    if ('round' in model_data):
        return model_data['round']

def save_lr_scheduler(lr_scheduler, round_idx, model_path):
    """ Save the current model weights.
    It includes: round and lr_scheduler_state

    Note that we must save the lr_scheduler to a temporal path + "_temp" first 
    to make sure the saving process is complete before renaming the target model path.

    Parameters:
        model_path (string, required): the path to save
    """
    make_dir(model_path)
    model_dict = {
        'round': round_idx,
        'lr_scheduler_state': lr_scheduler.state_dict(),
    }
    # save checkpoint
    torch.save(model_dict, model_path)