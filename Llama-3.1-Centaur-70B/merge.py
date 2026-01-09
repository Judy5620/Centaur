
import pandas as pd
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    df_test = pd.read_csv('results/' + args.model + '.csv', index_col=0)
    df_test['custom_metric'] = False
    df_test['unseen'] = 'participants'

    df_test_custom_metrics = pd.read_csv('results/custom_metrics_' + args.model + '.csv', index_col=0)
    df_test_custom_metrics['custom_metric'] = True
    df_test_custom_metrics['unseen'] = 'participants'

    df_generalization = pd.read_csv('generalization/results/' + args.model + '.csv', index_col=0)
    df_generalization['custom_metric'] = False
    df_generalization['unseen'] = 'experiments'

    df_generalization_custom_metrics = pd.read_csv('generalization/results/custom_metrics_' + args.model + '.csv', index_col=0)
    df_generalization_custom_metrics['custom_metric'] = True
    df_generalization_custom_metrics['unseen'] = 'experiments'

    df = pd.concat([df_test, df_test_custom_metrics, df_generalization, df_generalization_custom_metrics])
    print(df)

    df.to_csv('results/all_data_' + args.model.replace('/', '-') +  '.csv')
