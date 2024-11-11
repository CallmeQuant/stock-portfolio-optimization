from utils import load_obj
from Generative.FourierFlow import FourierFlow

def init_generator(config):
    """
    Initialize the FourierFlow generator and load its state from a checkpoint.

    Parameters:
    -----------
    config : object
        Configuration object containing model parameters and paths.

    Returns:
    --------
    generator : FourierFlow
        The initialized and loaded generator model.
    """
    print("Initialization of the model.")

    generator = FourierFlow(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden=config.hidden_dim,
        n_flows=config.n_flows,
        n_lags=config.n_lags,
        FFT=True,
        flip=True,
        normalize=False
    )

    print("Loading the model.")
    PATH_TO_MODEL = config.model_path 

    combined_state_dict = load_obj(PATH_TO_MODEL)

    # Check if the loaded object contains the 'generator' key
    if isinstance(combined_state_dict, dict) and 'generator' in combined_state_dict:
        generator.load_state_dict(combined_state_dict['generator'])
    else:
        generator.load_state_dict(combined_state_dict)

    generator.eval()

    return generator