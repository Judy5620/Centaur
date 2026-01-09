import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np
from transformers import pipeline
import torch
from sklearn.manifold import MDS
import seaborn as sns
import scienceplots


embeddings = []
colors = []

feature_extractor = pipeline("feature-extraction", framework="pt", model="answerdotai/ModernBERT-base")

# train data
dataset = load_dataset("marcelbinz/Psych-101")
unique_experiment_names = dataset.unique('experiment')['train']
for experiment_name in unique_experiment_names:
    print(experiment_name)
    subset = dataset.filter(lambda example: example["experiment"].startswith(experiment_name))
    text = subset['train'][0]['text'].split('<<')[0]
    features = feature_extractor(text, return_tensors = "pt").mean(axis=(0, 1))
    embeddings.append(features)
    colors.append('#69005f')

# eval data
eval_experiments = [
        "../generalization/dubois2022value/prompts.jsonl",
        "../generalization/feher2020humans/prompts.jsonl",
        "../generalization/jansen2021logic/prompts.jsonl",
        "../generalization/additional_experiments/awad2018moral.jsonl",
        "../generalization/additional_experiments/demircan2024evaluatingcategory.jsonl",
        "../generalization/additional_experiments/demircan2024evaluatingreward.jsonl",
        "../generalization/additional_experiments/akata2023repeatedgames.jsonl",
        "../generalization/additional_experiments/singh2022representing.jsonl",
        "../generalization/additional_experiments/xu2021novelty.jsonl",
]
eval_experiments_names = [
    'Modified problem structure (Figure 3b)',
    'Modified cover story (Figure 3a)',
    'Entirely novel domain (Figure 3c)',
    'Moral decision-making',
    'Naturalistic category learning',
    'Naturalistic reward learning',
    'Economic games',
    'Behavioral propensities',
    'Deep sequential decision task',
]

colors.extend(['C0', 'C1', 'C2', '#ff506e', '#ff506e', '#ff506e', '#ff506e', '#ff506e', '#ff506e'])
for eval_experiment_name in eval_experiments:
    subset = load_dataset('json',
        data_files={
            'train': [eval_experiment_name],
        }
    )
    text = subset['train'][0]['text'].split('<<')[0]
    features = feature_extractor(text, return_tensors = "pt").mean(axis=(0, 1))
    embeddings.append(features)

embeddings = torch.stack(embeddings, dim=0).numpy()
print(embeddings.shape)
reducer = MDS(n_components=2)
embeddings = reducer.fit_transform(embeddings)
print(embeddings.shape)

plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661, 7.08661/2))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, s=25, alpha=0.8)
plt.xlabel('Embedding dimension 1')
plt.ylabel('Embedding dimension 2')

for i in range(1, len(eval_experiments_names) + 1):
    plt.annotate(eval_experiments_names[-i], (0.2 + embeddings[-i, 0], embeddings[-i, 1]-0.2))

sns.despine()
plt.tight_layout()
plt.savefig('figures/fig12.pdf', bbox_inches='tight')
plt.show()
