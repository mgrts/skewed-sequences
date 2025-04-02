import typer

from skewed_sequences.generate_data import main as generate_data_main
from skewed_sequences.modeling.train import main as train_main


def run_all_experiments():
    # Define dataset-generation configurations
    dataset_configs = [
        {'lam': 0.0, 'q': 1.001, 'experiment_name': 'Transformer-SGT-heavy-tailed'},
        {'lam': 0.0, 'q': 9999.0, 'experiment_name': 'Transformer-SGT-normal'},
        {'lam': 0.9, 'q': 9999.0, 'experiment_name': 'Transformer-SGT-normal-skewed'},
        {'lam': 0.9, 'q': 1.001, 'experiment_name': 'Transformer-SGT-heavy-tailed-skewed'},
    ]

    # Define training configurations
    training_configs = [
        {'loss_type': 'sgt', 'sgt_lambda': 0.0,   'sgt_q': 1.001},
        {'loss_type': 'sgt', 'sgt_lambda': 0.0,   'sgt_q': 9999.0},
        {'loss_type': 'sgt', 'sgt_lambda': 0.001, 'sgt_q': 9999.0},
        {'loss_type': 'sgt', 'sgt_lambda': 0.001, 'sgt_q': 1.001},
        {'loss_type': 'mse'},
        {'loss_type': 'mae'},
    ]

    for ds_config in dataset_configs:
        lam = ds_config['lam']
        q = ds_config['q']
        base_experiment_name = ds_config['experiment_name']

        typer.echo(f'==== Generating dataset with lam={lam}, q={q} ====')

        generate_data_main(lam=lam, q=q)

        typer.echo('Dataset generation complete.\n')

        for train_config in training_configs:
            loss_type = train_config['loss_type']
            experiment_name = f'{base_experiment_name}'

            typer.echo(f'==== Starting training: {experiment_name} with loss_type={loss_type} ====')

            if loss_type.lower() == 'sgt':
                train_main(
                    loss_type=loss_type,
                    sgt_lambda=train_config['sgt_lambda'],
                    sgt_q=train_config['sgt_q'],
                    experiment_name=experiment_name,
                )
            else:
                train_main(
                    loss_type=loss_type,
                    experiment_name=experiment_name,
                )
            typer.echo(f'==== Completed training: {experiment_name} ====\n')


if __name__ == '__main__':
    run_all_experiments()
