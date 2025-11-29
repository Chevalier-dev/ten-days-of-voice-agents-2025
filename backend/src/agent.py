import logging
import os
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    tokenize,
    RoomInputOptions,
    MetricsCollectedEvent,
    metrics,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# =====================================================================
#   SIMPLE D&D-STYLE GAME MASTER AGENT  (DAY 8 PRIMARY GOAL)
# =====================================================================

class GameMasterAgent(Agent):
    def __init__(self):
        super().__init__(
    instructions="""
You are a D&D-style Game Master in the fantasy world of Eldoria.

RULES:
- Keep every response VERY short (1–2 sentences).
- No monologues, no paragraphs, no long descriptions.
- Always end with a simple question: "What do you do?"
- Maintain continuity based on the conversation.
- Keep pacing fast and interactive.

GAME START:
Before beginning the adventure, ask: "What is your character's name?"
After the player gives a name, start a short intro scene (1–2 sentences) and ask for their first action.

WORLD:
Eldoria is a high-fantasy realm of magic, ancient ruins, enchanted forests, mystical creatures, guilds, and hidden quests.
"""
        )

# =====================================================================
#   ENTRYPOINT (Same for all your agents)
# =====================================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    load_dotenv(".env.local", override=True)

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY missing!")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-anusha",
            style="Narration",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2)
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _m(ev: MetricsCollectedEvent):
        usage.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage Summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        )
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
