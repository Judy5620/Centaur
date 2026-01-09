import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from natsort import natsorted
import glob
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Ridge regression with specified parameters.')
    parser.add_argument('--model', type=str, default='Llama-3.1-Centaur-70B', help='centaur or Llama')
    parser.add_argument('--layer', type=int, help='From which layer of the LLM do you wanna extract representations from.')
    parser.add_argument('--roi', type=str, default='Left Accumbens', help='ROI to predict.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # process ROIs
    Y = pd.read_csv('results/parcels.csv')
    Y['block'] = pd.factorize(pd._libs.lib.fast_zip([Y.participant.values, Y.run_no.values]))[0]
    target_columns = ['participant', 'block', 'sub_trial_type'] + [f'X_{args.roi}']
    Y = Y[target_columns]

    # process representations
    feature_files = natsorted(glob.glob(f'results/*{args.model}*.pth'))
    X = [torch.load(file)[args.layer].numpy() for file in feature_files]

    # run models
    r2_scores = run(X, Y)
    torch.save(r2_scores, f"fits/model={args.model}_layer={args.layer}_roi={args.roi}.pth")

def run(X, Y):
    alphas = [10 ** c for c in range(0, 20)]
    participants = Y['participant'].unique()

    ### nested cross validation to identify optimal regularization values ###
    r2_scores = np.zeros((len(participants), 3, 3, len(alphas), 2, 3))
    for participant, participant_idx in enumerate(participants):
        print(f"Participant {participant}", flush=True)
        logo_outer = LeaveOneGroupOut()
        logo_inner = LeaveOneGroupOut()
        X_participant = X[participant]
        Y_participant = Y[Y['participant'] == participant_idx]

        for column in Y_participant.columns:
            if column.startswith('X_'):
                print(column, flush=True)
                Y_participant[column] = Y_participant.groupby('block', group_keys=False).apply(lambda g: (g[column] - g[column].mean()) / g[column].std())

        blocks_participant = Y_participant['block'].values
        Y_participant = Y_participant.drop(columns=['participant', 'block', 'sub_trial_type']).values

        print(X_participant.shape)
        print(Y_participant.shape)

        for fold_outer, (train_index_outer, test_index_outer) in enumerate(logo_outer.split(X_participant, Y_participant, blocks_participant)):
            X_train, X_test = X_participant[train_index_outer], X_participant[test_index_outer]
            Y_train, Y_test = Y_participant[train_index_outer], Y_participant[test_index_outer]
            blocks_train, blocks_test = blocks_participant[train_index_outer], blocks_participant[test_index_outer]

            for fold_inner, (train_index_inner, test_index_inner) in enumerate(logo_inner.split(X_train, Y_train, blocks_train)):
                X_train_inner, X_validation = X_train[train_index_inner], X_train[test_index_inner]
                Y_train_inner, Y_validation = Y_train[train_index_inner], Y_train[test_index_inner]
                blocks_train_inner, blocks_validation = blocks_train[train_index_inner], blocks_train[test_index_inner]

                scaler = StandardScaler()
                X_train_inner = scaler.fit_transform(X_train_inner)
                X_validation = scaler.transform(X_validation)
                X_test_pth = scaler.transform(X_test)

                pca = PCA(n_components=0.95)
                X_train_inner = pca.fit_transform(X_train_inner)
                X_validation = pca.transform(X_validation)
                X_test_pth = pca.transform(X_test_pth)
                print(pca.explained_variance_ratio_.sum(), flush=True)

                X_train_inner = torch.from_numpy(X_train_inner).float().to("cuda")
                X_validation = torch.from_numpy(X_validation).float().to("cuda")
                Y_train_inner = torch.from_numpy(Y_train_inner).float().to("cuda")
                Y_validation = torch.from_numpy(Y_validation).float().to("cuda")
                X_test_pth = torch.from_numpy(X_test_pth).float().to("cuda")
                Y_test_pth = torch.from_numpy(Y_test).float().to("cuda")


                A = X_train_inner.T @ X_train_inner
                I = torch.eye(A.shape[0]).to("cuda")
                c = X_train_inner.T @ Y_train_inner
                for alpha_idx, alpha in enumerate(alphas):
                    alpha_I = alpha * I
                    B = A + alpha_I
                    w = torch.linalg.solve(B, c)

                    Y_train_inner_pred = X_train_inner @ w
                    Y_validation_pred = X_validation @ w
                    Y_test_pred = X_test_pth @ w

                    r2_scores[participant, fold_outer, fold_inner, alpha_idx, :, 0] = r2_score(Y_train_inner.detach().cpu().numpy(), Y_train_inner_pred.detach().cpu().numpy(), multioutput="raw_values")
                    r2_scores[participant, fold_outer, fold_inner, alpha_idx, :, 1] = r2_score(Y_validation.detach().cpu().numpy(), Y_validation_pred.detach().cpu().numpy(), multioutput="raw_values")
                    r2_scores[participant, fold_outer, fold_inner, alpha_idx, :, 2] = r2_score(Y_test_pth.detach().cpu().numpy(), Y_test_pred.detach().cpu().numpy(), multioutput="raw_values")

    ### refit with optimal regularization values ###
    r2_scores_final = np.zeros((len(participants), 3, 2, 2))
    for participant, participant_idx in enumerate(participants):
        participant_mean_r2 = r2_scores[participant].mean((1, 3))
        participant_mean_r2_validation = participant_mean_r2[:, :, 1]
        best_alphas = participant_mean_r2_validation.argmax(1)

        print(best_alphas)
        print(f"Participant {participant}", flush=True)
        logo_outer = LeaveOneGroupOut()
        X_participant = X[participant]
        Y_participant = Y[Y['participant'] == participant_idx]

        for column in Y_participant.columns:
            if column.startswith('X_'):
                print(column, flush=True)
                Y_participant[column] = Y_participant.groupby('block', group_keys=False).apply(lambda g: (g[column] - g[column].mean()) / g[column].std())

        blocks_participant = Y_participant['block'].values
        Y_participant = Y_participant.drop(columns=['participant', 'block', 'sub_trial_type']).values

        print(X_participant.shape)
        print(Y_participant.shape)

        for fold_outer, (train_index_outer, test_index_outer) in enumerate(logo_outer.split(X_participant, Y_participant, blocks_participant)):
            X_train, X_test = X_participant[train_index_outer], X_participant[test_index_outer]
            Y_train, Y_test = Y_participant[train_index_outer], Y_participant[test_index_outer]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            pca = PCA(n_components=0.95)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            X_train = torch.from_numpy(X_train).float().to("cuda")
            Y_train = torch.from_numpy(Y_train).float().to("cuda")
            X_test = torch.from_numpy(X_test).float().to("cuda")
            Y_test = torch.from_numpy(Y_test).float().to("cuda")

            A = X_train.T @ X_train
            I = torch.eye(A.shape[0]).to("cuda")
            c = X_train.T @ Y_train

            alpha_I = alphas[best_alphas[fold_outer]] * I
            B = A + alpha_I
            w = torch.linalg.solve(B, c)

            Y_train_pred = X_train @ w
            Y_test_pred = X_test @ w

            r2_scores_final[participant, fold_outer, :, 0] = r2_score(Y_train.detach().cpu().numpy(), Y_train_pred.detach().cpu().numpy(), multioutput="raw_values")
            r2_scores_final[participant, fold_outer, :, 1] = r2_score(Y_test.detach().cpu().numpy(), Y_test_pred.detach().cpu().numpy(), multioutput="raw_values")

    return r2_scores_final

if __name__ == "__main__":
    main()
