import torch
import torch.nn as nn
from torch_geometric.utils import scatter

def get_loss_by_string(losses):
    '''
    Get loss function by string

    Parameters:
        losses (dict): The loss function settings.
            key (str): The loss function type.
            value (dict): The loss function settings.
                weight (float): The weight for the loss function. Default: None.
                mode (str): The loss function to use. Default: 'mse'.
                    'mse': Mean squared error.
                    'mae': Mean absolute error.
                    'huber': Huber loss.
                transform (str): The transformation to apply to the data. Default: None.
                    'cos': 1 - cosine similarity.
                    'norm': Norm.

    Returns:
        main_loss (nn.Module): The main loss function for model training (back propagation) and validation (learning rate scheduling).
        eval_loss (nn.Module): The evaluation loss function for task-specific model evaluation.
    '''
    main_losses = []
    eval_losses = []
    assert losses is not None, 'losses is not defined.'
    for key, kwargs in losses.items():
        if key == 'energy':
            main_losses.append(EnergyLoss(**kwargs))
            eval_losses.append(EnergyLoss(mode='mae'))
            eval_losses.append(EnergyLoss(mode='mse'))
            eval_losses.append(EnergyPerAtomLoss(mode='mae'))
            eval_losses.append(EnergyPerAtomLoss(mode='mse'))
        elif key == 'gradient_force':
            main_losses.append(GradientForceLoss(**kwargs))
            eval_losses.append(GradientForceLoss(mode='mae'))
            eval_losses.append(GradientForceLoss(mode='mse'))
        elif key == 'direct_force':
            main_losses.append(DirectForceLoss(**kwargs))
            eval_losses.append(DirectForceLoss(mode='mae'))
            eval_losses.append(DirectForceLoss(mode='mse'))
            eval_losses.append(DirectForceLoss(mode='mae', transform='cos'))
            eval_losses.append(DirectForceLoss(mode='mse', transform='cos'))
            eval_losses.append(DirectForceLoss(mode='mae', transform='norm'))
            eval_losses.append(DirectForceLoss(mode='mse', transform='norm'))
        elif key == 'dipole':
            main_losses.append(DipoleLoss(**kwargs))
            eval_losses += [
                DipoleLoss(mode='mae'),
                DipoleLoss(mode='mse'),
                DipoleLoss(mode='mae', transform='cos'),
                DipoleLoss(mode='mse', transform='cos'),
                DipoleLoss(mode='mae', transform='norm'),
                DipoleLoss(mode='mse', transform='norm'),
            ]
        elif key == 'charge':
            main_losses.append(ChargeLoss(**kwargs))
            eval_losses.append(ChargeLoss(mode='mae'))
            eval_losses.append(ChargeLoss(mode='mse'))
            eval_losses.append(ChargeLoss(mode='mae', transform='cos'))
            eval_losses.append(ChargeLoss(mode='mse', transform='cos'))
            eval_losses.append(ChargeLoss(mode='mae', transform='norm'))
            eval_losses.append(ChargeLoss(mode='mse', transform='norm'))


    def main_loss(pred, data):
        results = []
        for loss_fn in main_losses:
            try:
                out = loss_fn(pred, data)
            except Exception as e:
                print(f"\n❌ Error in loss_fn: {loss_fn.__class__.__name__}")
                print(f"type(pred): {type(pred)}")
                print(f"type(data): {type(data)}")
                # 打印 data 中每个键的类型
                if isinstance(data, dict):
                    for k, v in data.items():
                        print(f"  data['{k}']: {type(v)} -> {getattr(v, 'shape', None)}")
                # 打印预测结果中的每个键类型
                if isinstance(pred, dict):
                    for k, v in pred.items():
                        print(f"  pred['{k}']: {type(v)} -> {getattr(v, 'shape', None)}")
                raise e
            results.append(out)
        return sum(results)

    def eval_loss(pred, data):
        result_dict = {}
        for loss_fn in eval_losses:
            try:
                result_dict[loss_fn.name] = loss_fn(pred, data)
            except Exception as e:
                print(f"\n❌ Error in eval loss: {loss_fn.name}")
                print(f"type(pred): {type(pred)}")
                print(f"type(data): {type(data)}")
                if isinstance(data, dict):
                    for k, v in data.items():
                        print(f"  data['{k}']: {type(v)} -> {getattr(v, 'shape', None)}")
                if isinstance(pred, dict):
                    for k, v in pred.items():
                        print(f"  pred['{k}']: {type(v)} -> {getattr(v, 'shape', None)}")
                raise e
        return result_dict

    return main_loss, eval_loss

class BaseLoss(nn.Module):
    '''
    Base loss class

    Parameters:
        mode (str): The loss function to use.
            'mse': Mean squared error.
            'mae': Mean absolute error.
            'huber': Huber loss.
        weight (float): The weight for the loss function. Default: 1.
        transform (str): The transformation to apply to the data. Default: None.
            'cos': 1 - cosine similarity.
            'norm': Norm.
    '''
    def __init__(self, mode: str = 'mse', weight: float = 1, transform: str = None, **kwargs):
        super().__init__()
        self.weight = weight
        self.mode = mode
        if self.mode == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.mode == 'mae':
            self.loss_fn = nn.L1Loss()
        elif self.mode == 'huber':
            self.loss_fn = nn.HuberLoss(**kwargs)
        else:
            raise ValueError(f'loss mode {mode} not implemented')
        self.transform = transform
        if self.transform is None:
            pass
        elif self.transform == 'cos':
            self.cos = nn.CosineSimilarity(dim=-1)
            self.transform_fn = lambda x, y: (self.cos(x, y), torch.ones(x.shape[:-1], device=x.device))
        elif self.transform == 'norm':
            self.transform_fn = lambda x, y: (x.norm(dim=-1), y.norm(dim=-1))
        else:
            raise ValueError(f'transform {transform} not implemented')
        
    def forward(self, pred, data):
        if self.weight == 0:
            return 0
        value_pred, value_data = self.from_outputs(pred, data)
        if self.transform is not None:
            value_pred, value_data = self.transform_fn(value_pred, value_data)
        loss = self.loss_fn(value_pred, value_data)
        if self.weight == 1:
            return loss
        else:
            return self.weight * loss
    
    def from_outputs(self, pred, data):
        raise NotImplementedError
    

class EnergyLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.transform is None:
            self.name = f'energy_{self.mode}'
        else:
            raise ValueError(f'transform {self.transform} for energy not implemented')

    def from_outputs(self, pred, data):
        return pred.energy, data.energy

class EnergyPerAtomLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.transform is None:
            self.name = f'energy_per_atom_{self.mode}'
        else:
            raise ValueError(f'transform {self.transform} for energy per atom not implemented')

    def from_outputs(self, pred, data):
        n_atoms = scatter(torch.ones_like(data.z), data.batch)
        return pred.energy / n_atoms, data.energy / n_atoms

class GradientForceLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.transform is None:
            self.name = f'gradient_force_{self.mode}'
        else:
            self.name = f'gradient_force_{self.transform}_{self.mode}'
            
    def from_outputs(self, pred, data):
        return pred.gradient_force, data.force
    
class DirectForceLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.transform is None:
            self.name = f'direct_force_{self.mode}'
        else:
            self.name = f'direct_force_{self.transform}_{self.mode}'
            
    def from_outputs(self, pred, data):
        return pred.direct_force, data.force
        
class DipoleLoss(BaseLoss):
    '''
    Dipole moment loss

    Supports transform modes:
        - None: direct component-wise regression
        - 'cos': 1 - cosine similarity
        - 'norm': dipole magnitude comparison
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = f'dipole_{self.transform+"_" if self.transform else ""}{self.mode}'

    def from_outputs(self, pred, data):
        return pred.dipole, data.dipole
class ChargeLoss(BaseLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = f'charge_{self.transform+"_" if self.transform else ""}{self.mode}'

    def from_outputs(self, pred, data):
        return pred.charge, data.charge
