import os


USE_SQS = os.getenv("USE_SQS", "false").lower() == "true"

# Con eso decides usar redis o AWS cambiando una sola variable de entorno:
if USE_SQS:
    from .sqs_queue import send_message, receive_messages, delete_message
else:
    from .redis_queue import send_message, receive_messages, delete_message
