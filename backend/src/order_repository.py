import json
import os
from datetime import datetime

class OrderRepository:
    def __init__(self, path="orders"):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def save_order(self, order_data):
        order_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.path}/order_{order_id}.json"

        with open(filename, "w") as f:
            json.dump(order_data, f, indent=2)

        return filename
