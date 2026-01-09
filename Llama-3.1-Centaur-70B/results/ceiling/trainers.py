import torch
import torch.optim as optim
import torch.nn.functional as F
import schedulefree
from tqdm import tqdm

class Trainer:
    def __init__(self, model, num_iter=1000):
        self.model = model
        self.num_iter = num_iter
        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=0.1)

    def fit_and_evaluate(self, train_df, eval_df):
        ### PREPROCESS DATA ###
        train_data, eval_data = self.model.preprocess_data(train_df, eval_df)

        ### FITTING ###
        self.model.train()
        self.optimizer.train()
        for _ in tqdm(range(self.num_iter)):
            self.optimizer.zero_grad()
            logits = self.model(train_data)
            loss = F.cross_entropy(logits.flatten(0, -2), train_data['choice'].flatten().long())
            loss.backward()
            print(loss.item(), flush=True)
            self.optimizer.step()

        ### EVALUATION ###
        self.model.eval()
        self.optimizer.eval()
        logits = self.model(eval_data)
        return F.cross_entropy(logits.flatten(0, -2), eval_data['choice'].flatten().long())
