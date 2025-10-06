import numpy as np
import math
import torch
import torch.nn.functional as F
# Custom sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Vectorized sigmoid
sigmoid_v = np.vectorize(sigmoid)

def inference(model, dataloader, device):
    """
    Perform inference using a BERT model on a given dataloader.

    Args:
        model (torch.nn.Module): The trained BERT model.
        dataloader (torch.utils.data.DataLoader): DataLoader for test or validation data.
        device (torch.device): The device to run the inference on (e.g., 'cpu' or 'cuda').

    Returns:
        tuple: (list of probabilities, list of true labels)
    """
    # Set model to evaluation mode
    model.eval()

    # Tracking variables
    logits_list = []
    labels_list = []

    # Iterate through batches in dataloader
    for batch in dataloader:
        # Load batch to device
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Perform inference without gradient computation
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Append logits and labels to tracking variables
        logits_list.extend(logits.cpu().numpy())
        labels_list.extend(b_labels.cpu().numpy())

    # Calculate probabilities using sigmoid

    #probs = (np.sum(sigmoid_v(logits_list), axis=0).flatten() / len(logits_list))[1]
    probs = torch.softmax(torch.tensor(logits_list), dim=1)  
    probs2=[]
    for i in range(len(probs)):
        summed=np.array([1 if sublist[0] <= 0.55 else 0 for sublist in [probs[i]]]) #/len(probs[i])
        probs2.append(summed)
    score=(np.mean(probs2)) #* 100
    probs = torch.softmax(torch.tensor(logits_list), dim=1)
    # get pathogenic probabilities (second column)
    pathogenic_probs = probs[:, 1].numpy()
    # mean probability across chunks
    score = float(np.mean(pathogenic_probs))

    return score, labels_list,logits_list
