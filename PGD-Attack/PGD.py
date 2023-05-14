import time
import torch
import torch.nn as nn
import Util.load_dataset as DatasetLoader



def pgd(input, labels, model, iters, epsilon, stepsize, loss = None):
    input_copy = input.detach().clone() # Clones to create new set of adversarial images

    if(loss == None):
        loss = nn.CrossEntropyLoss()

    for _ in range(iters):
        input_copy.requires_grad = True
        model.zero_grad()
        pred = model(input_copy)
        loss_obj = loss(pred, y)
        loss_obj.backward()                      
        grad = input_copy.grad.detach()
        grad = grad.sign()
        input_copy = input_copy + stepsize * grad

        # Project x_copy onto x to get our adversarial x
        input_copy = input + torch.clamp(input_copy - input, min=-epsilon, max=epsilon)
        input_copy = input_copy.detach()
        input_copy = torch.clamp(input_copy, min=0, max=1)

    return input_copy








if __name__ == "__main__":
    # TODO: setup creation
    datasetLoader = DatasetLoader.LoadDataset()
    pgd()