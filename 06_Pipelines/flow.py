from typing import Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta

from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

from duration_prediction import read_dataframe, create_X, find_best_params, train_model


# ---- Task 1: download ----
@task(
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
)
def download(year: int, month: int):
    return read_dataframe(year=year, month=month)


# ---- Task 2: transform / prepare features ----
@task
def prepare_features(df_train, df_val):
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv=dv)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    return X_train, X_val, y_train, y_val, dv


# ---- Task 3: train and evaluate ----
@task(retries=0)
def train_and_evaluate(X_train, y_train, X_val, y_val, dv, max_evals: int = 50):
    best_params = find_best_params(X_train, y_train, X_val, y_val, max_evals=max_evals)
    run_id = train_model(X_train, y_train, X_val, y_val, dv, best_params)
    return run_id


# ---- Task 4: log / register ----
@task
def register_artifacts(run_id: str):
    # run_id.txt kept for any downstream shell step that still expects it;
    # the real channel is the flow's return value below.
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    return run_id


# ---- Orchestrator: compute the period ----
def resolve_periods(scheduled_date: datetime):
    """train = 2 months before scheduled_date, val = 1 month before."""
    train_date = scheduled_date - relativedelta(months=2)
    val_date = scheduled_date - relativedelta(months=1)
    return (train_date.year, train_date.month), (val_date.year, val_date.month)


# ---- Flow ----
@flow(name="nyc-taxi-training")
def train_flow(
    train_year: Optional[int] = None,
    train_month: Optional[int] = None,
    val_year: Optional[int] = None,
    val_month: Optional[int] = None,
    max_evals: int = 50,
):
    # If periods aren't passed explicitly, derive them from the flow's
    # own scheduled/logical run time -- not datetime.now() -- so a
    # backfill computes the right historical months.
    if train_year is None:
        from prefect.runtime import flow_run
        scheduled = flow_run.get_scheduled_start_time() or datetime.utcnow()
        (train_year, train_month), (val_year, val_month) = resolve_periods(scheduled)
    else:
        val_year = train_year if train_month < 12 else train_year + 1    
        val_month = train_month+ 1 if train_month < 12 else 1

    print(f"year: {train_year}, month: {train_month}")
    print(f"year_val: {val_year}, month_val: {val_month}")
    df_train = download(train_year, train_month)
    
    
    
    df_val = download(val_year, val_month)

    X_train, X_val, y_train, y_val, dv = prepare_features(df_train, df_val)

    run_id = train_and_evaluate(X_train, y_train, X_val, y_val, dv, max_evals=max_evals)

    final_run_id = register_artifacts(run_id)
    return final_run_id


if __name__ == "__main__":
    train_flow(train_year=2026, train_month=1)