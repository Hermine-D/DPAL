import torch
import torch.nn as nn

def count_parameters(model, unit='M'):
    """
    Count the total number of parameters, frozen parameters, and unfrozen parameters in the model.
    :param model: PyTorch model
    :param unit: Unit for parameter count: 'K' for thousands, 'M' for millions, 'G' for billions
    :return: Total, frozen, and unfrozen parameter counts
    """
    # Initialize counters
    total_params = 0
    frozen_params = 0
    unfrozen_params = 0

    # Iterate through all parameters
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params

        # Count frozen and unfrozen parameters
        if param.requires_grad:
            unfrozen_params += num_params
        else:
            frozen_params += num_params

    # Convert to the desired unit
    if unit == 'K':
        total_params /= 1000
        frozen_params /= 1000
        unfrozen_params /= 1000
    elif unit == 'M':
        total_params /= 1e6
        frozen_params /= 1e6
        unfrozen_params /= 1e6
    elif unit == 'G':
        total_params /= 1e9
        frozen_params /= 1e9
        unfrozen_params /= 1e9

    # Print the results
    print(f"Total parameters: {total_params:.2f} {unit}")
    print(f"Frozen parameters: {frozen_params:.2f} {unit}")
    print(f"Unfrozen parameters: {unfrozen_params:.2f} {unit}")