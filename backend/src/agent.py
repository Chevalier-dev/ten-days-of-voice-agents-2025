import logging

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
import os
import json
from datetime import datetime
from typing import List

def load_previous_checkins():
    """Load stored wellness check-ins from wellness_log.json."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_path = os.path.join(base_dir, "wellness_log.json")

    if not os.path.exists(log_path):
        return []

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

@function_tool
async def save_checkin(
    ctx: RunContext,
    mood: str,
    energy: str,
    goals: List[str],
    summary: str,
) -> str:
    """Save a daily health & wellness check-in to a JSON file."""

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_path = os.path.join(base_dir, "wellness_log.json")

    # Load existing log (or create new)
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "mood": mood,
        "energy": energy,
        "goals": goals,
        "summary": summary,
    }

    data.append(entry)

    # Persist file
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return "saved"


class Assistant(Agent):
    def __init__(self) -> None:
        previous_checkins = load_previous_checkins()
        memory_hint = ""

        if previous_checkins:
            last = previous_checkins[-1]
            memory_hint = (
                f"Last time, the user reported feeling '{last['mood']}' with energy '{last['energy']}'. "
                "Reference this gently when beginning today's check-in."
            )
        super().__init__(
            instructions=(
                "You are a daily health and wellness voice companion. "
                "When a new session begins, greet the user immediately and start a gentle check-in. "
                "Ask about mood, energy, and what they want to focus on today. "
                "Keep responses grounded, supportive, and non-clinical. "
                "\nYour job each session:\n"
                "- Ask how the user feels today.\n"
                "- Ask about energy levels.\n"
                "- Ask for 1–3 simple goals for the day.\n"
                "- Recap what they said.\n"
                "- Then call the tool save_checkin(mood, energy, goals, summary).\n"
                "After calling the tool, give a brief friendly closing message.\n"
                "If past check-ins exist, mention at least one insight (e.g., 'Yesterday you said your energy was low—how is it today?')."
            ),
            tools=[save_checkin],
        )
    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
