import os
import asyncio
import pika
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from categorization import categorize_text
from contextlib import asynccontextmanager

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

        # Publish the message
        channel.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key=queue_name,
            body=json.dumps(message),
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

            if not transcription_text:
                raise ValueError("Missing transcription text in message")

            # Categorize the transcription
            category = categorize_text(transcription_text)
            logger.info(f"Categorized transcription: {category}")

            # Send the categorized message to the next queue
            categorized_message = {"category": category}
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the RabbitMQ consumer."""
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
        category = categorize_text(transcription.transcription)
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
