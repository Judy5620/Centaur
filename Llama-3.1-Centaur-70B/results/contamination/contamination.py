from unsloth import FastLanguageModel
import transformers
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import torch
from scipy.optimize import minimize

# code from https://colab.research.google.com/drive/1GDbmEMmCVEOwhYk6-1AothdXeAlnqZ_j?usp=copy#scrollTo=L_mG7OJuumrZ
def get_logp(model,tokenizer,text):
    #get the logp for each token in text
    logp = []
    input_ids = tokenizer.encode(text)
    logits = model(torch.tensor(input_ids).unsqueeze(0)).logits
    logps = torch.nn.functional.log_softmax(logits[0,-len(input_ids):],dim=-1)
    for i in range(len(input_ids)-1):
        logp.append(logps[i,input_ids[i+1]].item())
    return [0]+logp

def fit_model(logp):
    # Fit an exponential model on a serie of logprobabilities (the function computes the cumulative probability)
    logp = np.array([0]+np.cumsum(np.array(logp)[1:]).tolist()) #Compute the cumulative logprobability
    def loss(logp, params):
        #Computes an MSE loss
        n = len(logp)
        A, B = params
        x = np.arange(len(logp))/n #Normalize x
        y = -A*(1-np.exp(-B*x))
        l = ((logp/n-y)**2).mean()
        return l
    #Fit the model
    A, B = minimize(
        lambda params : loss(logp,params),
        np.array([1,1]), #Arbitrary initialization
        method='BFGS',
        tol=10**-20).x
    return A, B

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
  max_seq_length = 32768,
  dtype = None,
  load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

dataset = load_dataset("marcelbinz/Psych-101")
unique_experiment_names = dataset.unique('experiment')['train']

As = []
Bs = []
with torch.no_grad():
    for experiment_name in unique_experiment_names:
        print(experiment_name)
        subset = dataset.filter(lambda example: example["experiment"].startswith(experiment_name))
        text = subset['train'][0]['text'].split('<<')[0]
        logp = get_logp(model,tokenizer,text)
        A, B = fit_model(logp)
        print('A:', A)
        print('B:', B)
        As.append(A)
        Bs.append(B)

torch.save(torch.Tensor(As), 'results/As.pth')
torch.save(torch.Tensor(Bs), 'results/Bs.pth')
