import sys
sys.path.append('.')
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
import pika
from MarketAgent.Market import StockData


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2020, 3, 14),
    "email": ["perusse.a@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}


dag = DAG(
    "gather_stock_data",
    default_args=default_args,
    schedule_interval=timedelta(1)
)

stocks = ['GOOGL', 'NVDA', 'APPL', 'TWTR', '.INX']


def get_queue_connection():
    credentials = pika.PlainCredentials('rabbit', 'rabbit')
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            'rabbitmq',
            5672,
            '/',
            credentials
        )
    )
    channel = connection.channel()
    return connection, channel


def create_queue(**kwargs):
    connection, channel = get_queue_connection()
    channel.queue_declare(queue='assets_to_query')
    connection.close()
    return 'Queue Created'


def add_asset_to_queue(asset, **kwargs):
    connection, channel = get_queue_connection()

    channel.basic_publish(
        exchange='',
        routing_key='assets_to_query',
        body=asset
    )
    connection.close()
    return f'Added {asset}'


def callback(ch, method, properties, body):
    print(body)
    series = gather_series(body)
    print(series)
    return series


def gather_series(asset, **kwargs):
    data = StockData(secret_path="./.secrets/alphavantage.key")
    data.key = "EQJ0ACQI8WG0LCNF"
    print(asset)
    series, meta = data.collect_data(
        asset=str(asset),
        interval='daily',
        output='full'
    )
    return series


def process_assets_in_queue(**kwargs):
    connection, channel = get_queue_connection()

    channel.basic_consume(
        queue='assets_to_query',
        on_message_callback=callback,
        auto_ack=True
    )

    channel.start_consuming()


with dag as dag:
    run_create_queue = PythonOperator(
        task_id='create_queue',
        provide_context=True,
        python_callable=create_queue
    )

    for asset in stocks:
        run_add_asset_to_queue = PythonOperator(
            task_id=f'add_{asset}_to_queue',
            provide_context=True,
            python_callable=add_asset_to_queue,
            op_kwargs={'asset': asset}
        )

        run_create_queue >> run_add_asset_to_queue

    run_process_assets_in_queue = PythonOperator(
            task_id=f'process_assets_in_queue',
            provide_context=True,
            python_callable=process_assets_in_queue
        )

    run_process_assets_in_queue << run_create_queue
