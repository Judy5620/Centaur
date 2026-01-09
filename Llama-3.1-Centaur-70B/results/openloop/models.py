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

class Temperature(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, values):
        """
        Multiplies the input tensor with a parameter.

        Parameter
        ---------
        values : tensor of any shape.

        Returns
        -------
        tensor of any shape
        Scaled tensor.
        """

        return values * self.beta

class Stickiness(nn.Module):
    def __init__(self, num_options):
        super().__init__()
        self.num_options = num_options
        self.beta = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, choices):
        """
        Returns whether a choice has been selected on the previous trial.

        Parameter
        ---------
        choices : tensor of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor filled with ones if choices has been selected on the previous trial.
        """

        num_tasks = choices.shape[0]

        previous_choices_0 = torch.zeros(num_tasks, 1, self.num_options)
        previous_choices_1 = torch.stack([(choices[:, :-1] == a).float() for a in range(self.num_options)], dim=-1)
        previous_choices = torch.cat([previous_choices_0, previous_choices_1], dim=1)

        return previous_choices * self.beta

class InformationBonus(nn.Module):
    def __init__(self, num_options):
        super().__init__()
        self.num_options = num_options
        self.beta = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, choices):
        """
        Returns how often a choice has been chosen up to trial t.

        Parameter
        ---------
        choices : tensor of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor containing how often a choice has been chosen up to trial t.
        """

        num_tasks = choices.shape[0]

        cumsum_choices_0 = torch.zeros(num_tasks, 1, self.num_options)
        cumsum_choices_1 = torch.stack([torch.cumsum((choices[:, :-1] == a).float(), dim=1) for a in range(self.num_options)], dim=-1)
        cumsum_choices = torch.cat([cumsum_choices_0, cumsum_choices_1], dim=1)

        return cumsum_choices * self.beta

class TabularRescorlaWagnerPlusMinusValueUpdating(nn.Module):
    def __init__(self, num_options, max_initial_values=100, ignore_index=-100):
        super().__init__()

        self.num_options = num_options
        self.max_initial_values = max_initial_values

        self.alpha_plus = nn.Parameter(0.01 * torch.randn([]))
        self.alpha_minus = nn.Parameter(0.01 * torch.randn([]))
        self.initial_values = nn.Parameter(0.01 * torch.randn([]))

        self.ignore_index = ignore_index

    def forward(self, choices, rewards):
        """
        Performs Rescorla-Wagner updating with separate learning rates for positive and negative prediction errors for the given choices and rewards.

        Parameter
        ---------
        choices : tensor of shape (N, T).
        rewards : tensor of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor filled with estimated values for all options.
        """

        num_tasks = choices.shape[0]
        num_trials = choices.shape[1]

        initial_values = self.max_initial_values * F.tanh(self.initial_values)
        alpha_plus = F.sigmoid(self.alpha_plus)
        alpha_minus = F.sigmoid(self.alpha_minus)

        values = torch.ones(list(choices.shape) + [self.num_options]) * initial_values

        for t in range(num_trials-1):
            # copy over everything
            values[:, t+1, :] = values[:, t, :]
            # compute prediction errors
            prediction_error = rewards[:, t] - values[torch.arange(num_tasks), t, choices[:, t]]
            # zero-out prediction errors for missing trial
            prediction_error[torch.isnan(rewards[:, t])] = 0
            # update values for selected actions
            values[torch.arange(num_tasks), t+1, choices[:, t]] = values[torch.arange(num_tasks), t, choices[:, t]] + (alpha_plus * prediction_error * (prediction_error >= 0).float()) + (alpha_minus * prediction_error * (prediction_error < 0).float())

        return values

class DualSystems(nn.Module):
    def __init__(self, variant, store_data=False):
        super().__init__()

        variants = ["two_step", "one_step"]
        assert variant in variants, f"variant must be one of {variants}"

        self.variant = variant
        self.store_data = store_data
        self.ignore_index = -100

        self.tau = nn.Parameter(torch.randn([]))
        self.alpha = nn.Parameter(torch.randn([]))
        self.lambd = nn.Parameter(torch.randn([]))
        self.stickiness = nn.Parameter(torch.randn([]))

        self.value_logits = Temperature()


    def preprocess_data(self, train_df, eval_df):

        train_df['choice'] = train_df['choice'].replace(2, -1)
        eval_df['choice'] = eval_df['choice'].replace(2, -1)
        train_df = train_df.replace(-1, np.nan)
        eval_df = eval_df.replace(-1, np.nan)

        for participant in train_df['participant'].unique():
            df_p = train_df[train_df['participant'] == participant]

        train_df_step1 = train_df[train_df['current_state'] == 999]
        train_df_step2 = train_df[train_df['current_state'] != 999]

        for participant in train_df_step1['participant'].unique():
            df_p1 = train_df_step1[train_df_step1['participant'] == participant]
            df_p2 = train_df_step2[train_df_step2['participant'] == participant]

        eval_df_step1 = eval_df[eval_df['current_state'] == 999]
        eval_df_step2 = eval_df[eval_df['current_state'] != 999]

        train_data1 = pd_to_pth(train_df_step1, ['current_state', 'reward', 'choice'], keys=['participant', 'trial'])
        train_data2 = pd_to_pth(train_df_step2, ['current_state', 'reward', 'choice'], keys=['participant', 'trial'])

        eval_data1 = pd_to_pth(eval_df_step1, ['current_state', 'reward', 'choice'], keys=['participant', 'trial'])
        eval_data2 = pd_to_pth(eval_df_step2, ['current_state', 'reward', 'choice'], keys=['participant', 'trial'])

        train_data = {}
        eval_data = {}
        for key in train_data1.keys():
            train_data[key] = torch.stack([train_data1[key], train_data2[key]], dim=-1)
            eval_data[key] = torch.stack([eval_data1[key], eval_data2[key]], dim=-1)

        train_data['choice'] = torch.nan_to_num(train_data['choice'], nan=self.ignore_index).long()
        eval_data['choice'] = torch.nan_to_num(eval_data['choice'], nan=self.ignore_index).long()

        return train_data, eval_data

    def forward(self, data):
        logits = self.forward_two_step(data) if self.variant == "two_step" else self.forward_one_step(data)
        return logits

    def forward_two_step(self, data):
        # TODO key presses not included atm

        if self.store_data:
            self.data = []

        tau = torch.sigmoid(self.tau)
        alpha = torch.sigmoid(self.alpha)
        lambd = torch.sigmoid(self.lambd)
        stickiness = torch.tanh(self.stickiness)

        action_1 = data['choice'][:, :, 0].long()
        action_2 = data['choice'][:, :, 1].long()
        state = data['current_state'][:, :, 1].long()
        reward = data['reward'][:, :, 1]

        transition_matrix = torch.Tensor([[0.7, 0.3], [0.3, 0.7]])
        n_participants = data['choice'].shape[0]
        n_trials = data['choice'].shape[1]
        action_repeat = torch.zeros(2)

        logits = torch.zeros(n_participants, n_trials, 2, 2)
        for par in range(n_participants):
            q_mf = torch.zeros(3, 2)  # initialise model free values
            for trial in range(n_trials):

                max_q, _ = torch.max(q_mf[1:], dim=1)
                q_mb = transition_matrix @ max_q
                q_net = tau * q_mb.clone() + (1 - tau) * q_mf[0].clone()

                if self.store_data:
                    self.data.append(np.concatenate([q_mf[0].detach().numpy(), q_mb.detach().numpy(), np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1)])) # step 1

                if (self.ignore_index == action_1[par, trial].item()): #, action_2[par, trial].item()
                    continue

                logits[par, trial, 0] = q_net + action_repeat * stickiness
                logits[par, trial, 1] = q_mf[state[par, trial] + 1]

                if self.store_data:
                    self.data.append(np.concatenate([q_mf[state[par, trial] + 1].detach().numpy(), q_mf[state[par, trial] + 1].detach().numpy(), np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1)])) # step 2

                if not (self.ignore_index == action_2[par, trial].item()):
                    delta_1 = q_mf[state[par, trial] + 1, action_2[par, trial]] - q_mf[0, action_1[par, trial]]
                    q_mf[0, action_1[par, trial]] += alpha * delta_1
                    delta_2 = reward[par, trial] - q_mf[state[par, trial] + 1, action_2[par, trial]]
                    q_mf[state[par, trial] + 1, action_2[par, trial]] += alpha * delta_2
                    q_mf[0, action_1[par, trial]] += lambd * alpha * delta_2

                    if self.store_data:
                        self.data.append(np.concatenate([0.5 * np.ones(2), 0.5 * np.ones(2), np.array([delta_1.item()]), np.array([delta_2.item()]), np.array([reward[par, trial]]), np.ones(1)])) # pe

                action_repeat = torch.zeros(2)
                action_repeat[action_1[par, trial]] = 1

        return self.value_logits(logits)

    def forward_one_step(self, data):
        return


class RescorlaWagnerModel(nn.Module):
    def __init__(self, num_options):
        super().__init__()

        self.num_options = num_options
        self.ignore_index = -100

        self.value_updating = TabularRescorlaWagnerPlusMinusValueUpdating(num_options)

        self.information_logits = InformationBonus(num_options)
        self.stickiness_logits = Stickiness(num_options)
        self.value_logits = Temperature()

    def preprocess_data(self, train_df, eval_df):
        """
        Preprocesses data into pytorch format.

        Parameter
        ---------
        train_df : pandas dataframe.
        eval_df : pandas dataframe.

        Returns
        -------
        dict, dict
        Dictionaries with tensors named 'choice' and 'reward' of shape (N, T).
        """
        if 'paper' in train_df.columns: # TODO fix for memory issues with wulff2018sampling
            selected_participants = np.random.choice(train_df['participant'].unique(), size=500, replace=False)
            train_df = train_df[train_df['participant'].isin(selected_participants)]

            #train_df = train_df.head(200000)
        print(len(train_df))
        print(len(eval_df))

        if 'forced' in train_df:
            train_data = pd_to_pth(train_df, ['reward', 'choice', 'forced'])
        else:
            train_data = pd_to_pth(train_df, ['reward', 'choice'])

        if 'forced' in train_df:
            eval_data = pd_to_pth(eval_df, ['reward', 'choice', 'forced'])
        else:
            eval_data = pd_to_pth(eval_df, ['reward', 'choice'])

        # deal with nans
        #assert (torch.isnan(train_data['reward']) == torch.isnan(train_data['choice'])).all()
        #assert (torch.isnan(eval_data['reward']) == torch.isnan(eval_data['choice'])).all()
        train_data['choice'] = torch.nan_to_num(train_data['choice'], nan=self.ignore_index).long()
        eval_data['choice'] = torch.nan_to_num(eval_data['choice'], nan=self.ignore_index).long()

        # store copy used for updating
        train_data['choice_for_updating'] = train_data['choice'].clone().clamp(min=0)
        eval_data['choice_for_updating'] = eval_data['choice'].clone().clamp(min=0)
        # if forced, don't use for loss computation
        if 'forced' in train_df:
            train_data['choice'][torch.nan_to_num(train_data['forced'], nan=1).long()] = self.ignore_index
        if 'forced' in train_df:
            eval_data['choice'][torch.nan_to_num(eval_data['forced'], nan=1).long()] = self.ignore_index

        return train_data, eval_data

    def forward(self, data):
        """
        Model with Rescorla-Wagner-based learning, stickiness and information bonus.

        Parameter
        ---------
        data : dictionary with tensors named 'choice' and 'reward' of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor filled with logits for all options.
        """

        values = self.value_updating(data['choice_for_updating'].long(), data['reward'].float())

        information_logits = self.information_logits(data['choice'].long())
        stickiness_logits = self.stickiness_logits(data['choice'].long())
        value_logits = self.value_logits(values)
        return value_logits + stickiness_logits + information_logits
