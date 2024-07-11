
def str_to_float_list(arg):
    return [float(x) for x in arg.strip('[]').split(',')]

def add_dataset_args(parser):
    
    parser.add_argument("--env", type=str, required=True, help="Environment")
    
    parser.add_argument("--Bustotal", type=int, required=False,
                        default=100, help="Total number of buses")

    parser.add_argument("--beta", type=float, required=False,
                        default=0.95, help="Beta")
    parser.add_argument("--theta", type=str_to_float_list, required=False, default=[1, 2, 9], help="Theta values as a list of floats")
    
    parser.add_argument("--H", type=int, required=False,
                        default=100, help="Context horizon")
    
    parser.add_argument("--maxMileage", type=int, required=False,
                        default=200, help="Max mileage")
    parser.add_argument("--numTypes", type=int, required=False,
                        default=10, help="Number of bus types")
    parser.add_argument("--extrapolation", type=str, required=False,
                        default='False', help="Extrapolation")
    


def add_model_args(parser):
    parser.add_argument("--embd", type=int, required=False,
                        default=32, help="Embedding size")
    parser.add_argument("--head", type=int, required=False,
                        default=1, help="Number of heads")
    parser.add_argument("--layer", type=int, required=False,
                        default=3, help="Number of layers")
    parser.add_argument("--lr", type=float, required=False,
                        default=1e-3, help="Learning Rate")
    parser.add_argument("--dropout", type=float,
                        required=False, default=0, help="Dropout")
    parser.add_argument('--shuffle', default=False, action='store_true')


def add_train_args(parser):
    parser.add_argument("--num_epochs", type=int, required=False,
                        default=500, help="Number of epochs")


def add_eval_args(parser):
    parser.add_argument("--epoch", type=int, required=False,
                        default=-1, help="Epoch to evaluate")
    parser.add_argument("--hor", type=int, required=False,
                        default=-1, help="Episode horizon (for mdp)")
    parser.add_argument("--n_eval", type=int, required=False,
                        default=100, help="Number of eval trajectories")
