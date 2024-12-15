import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import pika
import psycopg2  # For connecting to PostgreSQL
from fastapi import FastAPI, HTTPException
from psycopg2.extras import execute_values
from pydantic import BaseModel

from categorization import categorize_text_with_tags_and_category

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RabbitMQ connection settings
RABBITMQ_HOST = os.getenv('SPRING_RABBITMQ_HOST')
RABBITMQ_PORT = os.getenv('SPRING_RABBITMQ_PORT')
RABBITMQ_USERNAME = os.getenv('SPRING_RABBITMQ_USERNAME')
RABBITMQ_PASSWORD = os.getenv('SPRING_RABBITMQ_PASSWORD')
TRANSCRIPTION_QUEUE = "transcription_queue"
CATEGORIZATION_QUEUE = "categorization_queue"
EXCHANGE_NAME = "categorization_exchange"

DB_HOST = os.getenv("DB_HOST", "categorization_service_db_postgres")
DB_NAME = os.getenv("DB_NAME", "categorization_service_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


class TranscriptionRequest(BaseModel):
    transcription: str


def get_rabbitmq_connection():
    """Establish a RabbitMQ connection."""
    credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
    return pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_HOST, port=int(RABBITMQ_PORT), credentials=credentials
        )
    )


def send_to_queue(queue_name: str, message: dict):
    """Send a categorized message to the next queue."""
    try:
        connection = get_rabbitmq_connection()
        channel = connection.channel()

        # Declare the exchange and queue
        channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type="direct", durable=True)
        channel.queue_declare(queue=queue_name, durable=True)
        channel.queue_bind(exchange=EXCHANGE_NAME, queue=queue_name, routing_key=queue_name)

        # Publish the message with `ensure_ascii=False` to prevent escaping non-ASCII characters
        channel.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key=queue_name,
            body=json.dumps(message, ensure_ascii=False),  # Avoid escaping non-ASCII characters
            properties=pika.BasicProperties(delivery_mode=2),  # Make message persistent
        )
        logger.info(f"Message sent to queue '{queue_name}': {message}")

        connection.close()
    except Exception as e:
        logger.error(f"Failed to send message to queue '{queue_name}': {e}")
        raise e


async def consume_messages():
    """Consume messages from the transcription_queue."""
    connection = get_rabbitmq_connection()
    channel = connection.channel()
    channel.queue_declare(queue=TRANSCRIPTION_QUEUE, durable=True)

    def callback(ch, method, properties, body):
        try:
            # Decode the message
            message_data = json.loads(body.decode("utf-8"))
            transcription_text = message_data.get("transcription")
            transcription_tags = message_data.get("tags")
            transcription_category = message_data.get("category")
            channel_id = message_data.get("channelId")
            video_id = message_data.get("videoId")
            audio_part = message_data.get("audioPart")

            if not transcription_text:
                raise ValueError("Missing transcription text in message")

            # Categorize the transcription
            categorization_result = categorize_text_with_tags_and_category(transcription_text, tags=transcription_tags,
                                                                           category=transcription_category)

            store_categorization(categorization_result, channel_id, video_id, audio_part)

            # Send the categorized message to the next queue
            categorized_message = {"categorization_result": categorization_result, "channelId": channel_id,
                                   "videoId": video_id, "audio_part": audio_part, "transcription": transcription_text}
            send_to_queue(CATEGORIZATION_QUEUE, categorized_message)
            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    channel.basic_consume(queue=TRANSCRIPTION_QUEUE, on_message_callback=callback, auto_ack=False)
    logger.info("Started consuming messages from RabbitMQ transcription_queue")
    await asyncio.get_event_loop().run_in_executor(None, channel.start_consuming)
    connection.close()


def create_tables_if_not_exist():
    """Create the required database tables if they do not exist."""
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = connection.cursor()

        # SQL for creating tables
        create_categorization_table = """
        CREATE TABLE IF NOT EXISTS categorization (
            id SERIAL PRIMARY KEY,
            channel_id VARCHAR(255) NOT NULL,
            video_id VARCHAR(255) NOT NULL,
            audio_part VARCHAR(255),
            created_at TIMESTAMP DEFAULT NOW()
        );
        """

        create_category_table = """
        CREATE TABLE IF NOT EXISTS category (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE
        );
        """

        create_categorization_category_table = """
        CREATE TABLE IF NOT EXISTS categorization_category (
            id SERIAL PRIMARY KEY,
            categorization_id INT NOT NULL REFERENCES categorization(id) ON DELETE CASCADE,
            category_id INT NOT NULL REFERENCES category(id) ON DELETE CASCADE
        );
        """

        # Execute table creation queries
        cursor.execute(create_categorization_table)
        cursor.execute(create_category_table)
        cursor.execute(create_categorization_category_table)

        # Commit changes
        connection.commit()
        logger.info("Tables ensured to exist in the database.")

    except Exception as e:
        logger.info(f"Failed to create tables: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()


def store_categorization(categorization_result, channel_id, video_id, audio_part):
    """Store categorization results in the PostgreSQL database."""
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = connection.cursor()

        # Step 1: Insert into `categorization`
        cursor.execute(
            """
            INSERT INTO categorization (channel_id, video_id, audio_part)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (channel_id, video_id, audio_part)
        )
        categorization_id = cursor.fetchone()[0]

        # Step 2: Insert into `category` and fetch category IDs
        category_ids = []
        for categorization in categorization_result:
            category_name = categorization.get("category")
            if category_name:
                cursor.execute(
                    """
                    INSERT INTO category (name)
                    VALUES (%s)
                    ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                    RETURNING id;
                    """,
                    (category_name,)
                )
                category_ids.append(cursor.fetchone()[0])

        # Step 3: Insert into `categorization_category`
        categorization_category_values = [
            (categorization_id, category_id) for category_id in category_ids
        ]
        execute_values(
            cursor,
            """
            INSERT INTO categorization_category (categorization_id, category_id)
            VALUES %s;
            """,
            categorization_category_values
        )

        # Commit the transaction
        connection.commit()
        logging.info("Data stored successfully in categorization service DB.")

    except Exception as e:
        logging.error(f"Failed to store data: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the RabbitMQ consumer."""
    create_tables_if_not_exist()
    consumer_task = asyncio.create_task(consume_messages())
    yield
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


# FastAPI setup with lifespan
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Categorization Service is running"}


@app.post("/categorize")
async def categorize_text_request(transcription: TranscriptionRequest):
    """Categorize text via HTTP POST."""
    try:
        category = categorize_text_with_tags_and_category(transcription.transcription)
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
