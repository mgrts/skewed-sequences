import typer
from skewed_sequences.modeling.train import main as train_main


SGT_LOSS_LAMBDAS = [-0.1, -0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01, 0.1]


def run_experiments():
    """
    Runs training experiments for each value in SGT_LOSS_LAMBDAS.
    """
    for sgt_lambda in SGT_LOSS_LAMBDAS:
        typer.echo(f"Starting training with sgt_lambda = {sgt_lambda}")
        # Call the main training function with the current sgt_lambda
        train_main(sgt_lambda=sgt_lambda)
        typer.echo(f"Completed training with sgt_lambda = {sgt_lambda}\n")


if __name__ == "__main__":
    run_experiments()
