import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    torch.manual_seed(worker_seed)
    numpy_seed = int(worker_seed % (2**32 - 1))  # Optional, in case you also use numpy in the DataLoader
    np.random.seed(numpy_seed)





def build_Zurcher_data_filename(env, config, mode):
    """
    Builds the filename for the Zurcher data.
    Mode is either 0: train, 1: test, 2: eval.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = env
    if mode != 2:
        filename += '_bustotal' + str(config['bustotal'])
    filename += '_beta' + str(config['beta'])
    filename += '_theta' + str(config['theta'])    
    filename += '_numTypes' + str(config['numTypes'])    
    filename += '_H' + str(config['horizon'])
    filename += '_' + config['rollin_type']
    #filename += '_extrapolation' + str(config['extrapolation'])
    if mode == 0:
        filename += '_train'
    elif mode == 1:
        filename += '_test'
    elif mode == 2:
        filename += '_n_eval' + str(config['n_eval'])
        filename += '_eval'
        
    return filename_template.format(filename)



def build_Zurcher_model_filename(env, config):
    """
    Builds the filename for the Zurcher model.
    """
    filename = env
    filename += '_shuf' + str(config['shuffle'])
    filename += '_lr' + str(config['lr'])
    filename += '_do' + str(config['dropout'])
    filename += '_embd' + str(config['n_embd'])
    filename += '_layer' + str(config['n_layer'])
    filename += '_head' + str(config['n_head'])
    #filename += '_Bustotal' + str(config['Bustotal'])
    #filename += '_beta' + str(config['beta'])
    #filename += '_theta' + str(config['theta'])
    #filename += '_numTypes' + str(config['numTypes'])
    filename += '_H' + str(config['horizon'])
    filename += '_seed' + str(config['seed'])
    return filename




def convert_to_tensor(x, store_gpu=True):
    if store_gpu:
        return torch.tensor(np.asarray(x)).float().to(device)
    else:
        return torch.tensor(np.asarray(x)).float()