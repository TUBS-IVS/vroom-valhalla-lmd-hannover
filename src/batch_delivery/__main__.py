"""``python -m batch_delivery`` — module entry that delegates to the Typer CLI."""
from batch_delivery.cli import app

if __name__ == "__main__":
    app()
