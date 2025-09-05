from __future__ import annotations
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from ml.training import retrain_from_mongo
from ml.registry import promote_best
from extract import fetch_weather_data
from transform import transform_weather_data
from load import save_to_csv, save_to_mongodb
from config import MONGODB_PASSWORD

scheduler: BackgroundScheduler | None = None


def start_scheduler():
    global scheduler
    if scheduler is not None:
        return scheduler
    scheduler = BackgroundScheduler()

    def retrain_and_promote():
        try:
            # 1) Refresh data in MongoDB via ETL
            try:
                raw = fetch_weather_data()
                if raw:
                    df = transform_weather_data(raw)
                    try:
                        save_to_csv(df)
                    except Exception:
                        pass
                    save_to_mongodb(df, MONGODB_PASSWORD)
            except Exception as etl_err:
                print(f"ETL refresh failed (continuing to retrain on existing data): {etl_err}")
            # 2) Retrain from latest Mongo data
            retrain_from_mongo()
            # Try promoting the best of latest runs
            try:
                promote_best('regression')
            except Exception:
                pass
            try:
                promote_best('classification')
            except Exception:
                pass
        except Exception as e:
            print(f"Retrain job failed: {e}")

    # Retrain and promote every 5 minutes
    scheduler.add_job(retrain_and_promote, 'interval', minutes=5, id='retrain_every_5m', replace_existing=True)
    scheduler.start()
    print(f"APScheduler started at {datetime.now().isoformat()} (interval: every 5 minutes)")
    return scheduler


def stop_scheduler():
    global scheduler
    if scheduler:
        scheduler.shutdown(wait=False)
        scheduler = None
