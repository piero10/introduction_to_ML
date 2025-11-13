
import mlflow.sklearn

with mlflow.start_run() as run:
    print(f"Run ID: {run.info.run_id}")