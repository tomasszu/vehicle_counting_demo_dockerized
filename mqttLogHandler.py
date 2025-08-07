import logging
import paho.mqtt.client as mqtt

class MqttLogHandler(logging.Handler):
    def __init__(self, broker_host, broker_port, topic):
        super().__init__()
        self.topic = topic
        self.client = mqtt.Client()
        self.client.connect(broker_host, broker_port)
        self.client.loop_start()

    def emit(self, record):
        log_entry = self.format(record)
        self.client.publish(self.topic, log_entry)

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()
        super().close()
