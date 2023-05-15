import time
import torch
import torch.nn as nn
from tqdm import tqdm


def pgd(input, labels, model, niter, epsilon, stepsize, loss = None, randinit = False):
    """
    param input: a clean sample (or batch of samples)
    param labels: the labels of input
    param model: a pre-trained DNN you're attacking
    param niter: # of PGD iterations
    param epsilon: l-inf epsilon bound
    param stepsize: the step-size for PGD
    param loss: a loss function
    param randinit: start from a random perturbation if set true
    """
    input_copy = input.detach().clone() # Clones to create new set of adversarial images

    if(loss == None):
        loss = nn.CrossEntropyLoss()

    if randinit:
        random_perturbations_up_or_down = 2 * torch.rand(input.size(), dtype=input.dtype, device=input.device) - 1   # creates a tensor of perturbations in range (-1, 1)
        input = input + random_perturbations_up_or_down * epsilon   # bound the perturbations range to (-epsilon, epsilon) and apply them to x
        input = torch.clamp(input, min=0, max=1)    # make sure x is within range 0 to 1 after applying perturbations

    for _ in tqdm(range(niter)):
        input_copy.requires_grad = True
        pred = model(input_copy)
        loss_obj = loss(pred[1], labels)
        loss_obj.backward()                      
        grad = input_copy.grad.detach()
        grad = grad.sign()
        input_copy = input_copy + stepsize * grad

        input_copy = input + torch.clamp(input_copy - input, min=-epsilon, max=epsilon)
        input_copy = input_copy.detach()
        input_copy = torch.clamp(input_copy, min=0, max=1)

    return input_copy
