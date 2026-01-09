import ast
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def pd_to_pth(df, values, keys=['participant', 'task', 'trial']):
    column_names_list = [keys + [value] for value in values]
    wide_arrs = {}
    for column_names in column_names_list:
        arr = df[column_names].values
        dims = [np.unique(arr[:, i], return_inverse=True) for i in range(len(column_names)-1)]
        wide_arr = np.full([len(dims[i][0]) for i in range(len(column_names)-1)], np.nan)
        wide_arr[*[dims[i][1] for i in range(len(column_names)-1)]] = arr[:, -1]
        wide_arrs[column_names[-1]] = torch.from_numpy(wide_arr).reshape(-1, wide_arr.shape[-1])
    return wide_arrs

class DunningKruger(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_tensor = nn.Parameter(torch.randn(28, 11))

    def preprocess_data(self, train_df, eval_df):
        for i in range(4, 24):
            train_df.loc[train_df['trial'] == i, 'choice'] = train_df[train_df['trial'] == i]['choice'].astype('category').cat.codes
            eval_df.loc[eval_df['trial'] == i, 'choice'] = eval_df[eval_df['trial'] == i]['choice'].astype('category').cat.codes

        normalizer = torch.Tensor([2, 10, 1, 1,
                                   1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1,
                                   2, 10, 1, 1])

        train_data = {}
        num_train_participants = len(train_df.participant.unique())
        train_data['choice'] = torch.from_numpy(train_df[(train_df['trial'] != 24)]['choice'].values.astype('float'))
        train_data['choice'] = (train_data['choice'] // normalizer.repeat(num_train_participants)).long()

        eval_data = {}
        num_eval_participants = len(eval_df.participant.unique())
        eval_data['choice'] = torch.from_numpy(eval_df[(eval_df['trial'] != 24)]['choice'].values.astype('float'))
        eval_data['choice'] = (eval_data['choice'] // normalizer.repeat(num_eval_participants)).long()

        return train_data, eval_data

    def forward(self, data):
        num_participants = int(data['choice'].shape[0] / 28)
        params = self.param_tensor.repeat(num_participants, 1)

        return params

class NoiseCeiling(nn.Module):
    def __init__(self, UID, num_questions=14568, num_options=2, ):
        super().__init__()
        self.param_tensor = nn.Parameter(torch.randn(num_questions, num_options))
        self.UID = UID

    def preprocess_data(self, train_df, eval_df):
        train_data = {}

        mapping_dict = {k: v for k, v in zip(train_df[self.UID], train_df[self.UID].astype('category').cat.codes)}
        print(mapping_dict)

        train_data['choice'] = torch.from_numpy(train_df['choice'].values)
        train_data[self.UID] = torch.from_numpy(train_df[self.UID].map(mapping_dict).values).long()


        eval_data = {}
        num_eval_participants = len(eval_df.participant.unique())
        eval_data['choice'] = torch.from_numpy(eval_df['choice'].values)
        eval_data[self.UID] = torch.from_numpy(eval_df[self.UID].map(mapping_dict).values).long()

        return train_data, eval_data

    def forward(self, data):
        params = self.param_tensor[data[self.UID]]
        return params
