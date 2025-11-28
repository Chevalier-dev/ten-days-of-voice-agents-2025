import logging
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel


logger = logging.getLogger("agent")
load_dotenv(".env.local")


# ============================================================
#   Order Repository (JSON storage)
# ============================================================

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


# ============================================================
#   Food Ordering Agent (ONLY agent)
# ============================================================

class FoodOrderingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are QuickBasket — a friendly grocery & food ordering assistant.

Your job:
- Add/remove/update cart items using the proper tool.
- Ask for quantity if not provided.
- Interpret requests like “Ingredients for X” using add_recipe_items.
- When the user says “Place my order”, call place_order.
- Always confirm cart updates verbally.
- Never assume an item exists — check using the catalog.
"""
        )

        # Load catalog
        with open("catalog/catalog.json") as f:
            self.catalog = json.load(f)["items"]

        # Load recipes
        with open("catalog/recipes.json") as f:
            self.recipes = json.load(f)

        # Memory state
        self.cart = {}
        self.repo = OrderRepository()

    # -----------------------------------------------------
    # CART TOOLS
    # -----------------------------------------------------

    @function_tool
    async def add_to_cart(self, ctx: RunContext, item: str, quantity: int = 1):
        it = item.lower()
        match = next((i for i in self.catalog if i["name"].lower() == it), None)

        if not match:
            return f"Sorry, I don't have {item} in the catalog."

        self.cart[it] = self.cart.get(it, 0) + quantity
        return f"Added {quantity} {match['name']} to your cart."

    @function_tool
    async def remove_from_cart(self, ctx: RunContext, item: str):
        it = item.lower()
        if it in self.cart:
            del self.cart[it]
            return f"Removed {item} from your cart."
        return f"{item} is not in your cart."

    @function_tool
    async def list_cart(self, ctx: RunContext):
        if not self.cart:
            return "Your cart is currently empty."

        response = "Your cart:\n"
        for name, qty in self.cart.items():
            response += f"- {name} x {qty}\n"
        return response

    @function_tool
    async def add_recipe_items(self, ctx: RunContext, recipe: str):
        recipe = recipe.lower()
        if recipe not in self.recipes:
            return f"Sorry, I don’t know the ingredients for {recipe}."

        added_items = []
        for item in self.recipes[recipe]:
            it = item.lower()
            self.cart[it] = self.cart.get(it, 0) + 1
            added_items.append(item)

        return f"Added ingredients for {recipe}: {', '.join(added_items)}"

    # -----------------------------------------------------
    # ORDER TOOL
    # -----------------------------------------------------

    @function_tool
    async def place_order(self, ctx: RunContext, customer_name: str = "Customer"):
        if not self.cart:
            return "Your cart is empty — nothing to place!"

        items = []
        total = 0

        for name, qty in self.cart.items():
            info = next(i for i in self.catalog if i["name"].lower() == name)
            cost = info["price"] * qty
            items.append({"name": info["name"], "qty": qty, "price": cost})
            total += cost

        order_data = {
            "customer": customer_name,
            "items": items,
            "total": total,
            "timestamp": datetime.now().isoformat()
        }

        filename = self.repo.save_order(order_data)

        # Reset cart
        self.cart = {}

        return f"Your order has been placed! Saved to {filename}."


# ============================================================
#   ENTRYPOINT
# ============================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    load_dotenv(".env.local", override=True)

    # LLM key check
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not set")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-anusha",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2)
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_m(ev: MetricsCollectedEvent):
        usage.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage Summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=FoodOrderingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
