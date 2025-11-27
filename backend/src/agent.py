import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

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

# ------------------------------
# Logging
# ------------------------------

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# ============================================================
#  SDR ASSISTANT (YOUR EXISTING DAY 1–5 LOGIC)
# ============================================================

class SimpleSDRAssistant(Agent):
    def __init__(self) -> None:
        self.company_data = self._load_company_data()
        self.personas_data = self._load_personas_data()
        self.calendar_data = self._load_calendar_data()
        self.lead_data = {}
        self.conversation_transcript = []
        self.detected_persona = None
        self.conversation_ended = False
        
        super().__init__(
            instructions=f"""You are Nikita, SDR for {self.company_data['company']['name']}. Be CONCISE and professional.

MANDATORY OPENING SEQUENCE (ALWAYS DO THIS FIRST):
1. Greet: "Hi! I'm Nikita from Razorpay. Before we start, I need a few quick details."
2. Ask for NAME: "What's your name?"
3. Ask for EMAIL: "What's your email address?"
4. Ask for COMPANY: "Which company are you from?"
5. Ask for ROLE: "What's your role there?"
6. Ask for TEAM SIZE: "How big is your team?"
7. Ask for TIMELINE: "When are you looking to implement this - now, soon, or later?"
8. Ask for NEED: "What brings you to Razorpay today? What are you looking for?"

[...REST OF YOUR SDR PROMPT REMAINS EXACTLY SAME...]
""",
        )
    
    # === Data Loaders ===
    def _load_company_data(self) -> Dict[str, Any]:
        try:
            with open("company_data/razorpay_faq.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Company FAQ data not found")
            return {"company": {"name": "Razorpay"}, "faq": []}
    
    def _load_personas_data(self) -> Dict[str, Any]:
        try:
            with open("personas.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Personas data not found")
            return {"personas": {}}
    
    def _load_calendar_data(self) -> Dict[str, Any]:
        try:
            with open("mock_calendar.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Calendar data not found")
            return {"available_slots": [], "booked_meetings": []}

    # === All your same tools (unchanged) ===
    # I’m keeping the complete content exactly as you shared.

    @function_tool
    async def detect_persona(self, context: RunContext, user_input: str) -> str:
        input_lower = user_input.lower()
        
        persona_scores = {}
        for persona_name, persona_data in self.personas_data.get("personas", {}).items():
            score = 0
            for keyword in persona_data.get("keywords", []):
                if keyword in input_lower:
                    score += 1
            persona_scores[persona_name] = score
        
        if persona_scores:
            self.detected_persona = max(persona_scores, key=persona_scores.get)
            if persona_scores[self.detected_persona] > 0:
                self.lead_data["detected_persona"] = self.detected_persona
                return f"Got it! As a {self.detected_persona}, I can share how Razorpay helps people in your role."
        
        return "Thanks! Let me understand your needs better."

    @function_tool
    async def show_available_meetings(self, context: RunContext, meeting_type: str = "demo") -> str:
        available_slots = []
        
        for slot in self.calendar_data.get("available_slots", []):
            if slot.get("available", False) and slot.get("type") == meeting_type:
                available_slots.append(slot)
        
        if not available_slots:
            return "No slots available now. Try later?"

        options = available_slots[:5]
        name = self.lead_data.get("name", "")
        greeting = f"Great {name}! " if name else ""
        response = f"{greeting}Here are some available times:\n\n"
        
        for i, slot in enumerate(options, 1):
            response += f"{i}. {slot['date']} at {slot['time']} ({slot['duration']})\n"
        
        response += "\nWhich slot works for you?"
        return response

    @function_tool
    async def book_meeting(self, context: RunContext, slot_choice: str, meeting_type: str = "demo") -> str:
        if not self.lead_data.get("email"):
            return "I need your email first to send confirmation. What's your email?"

        available_slots = [s for s in self.calendar_data.get("available_slots", []) 
                          if s.get("available", False) and s.get("type") == meeting_type]

        if not available_slots:
            return "No slots available at the moment."

        selected_slot = None
        
        try:
            choice_num = int(slot_choice.strip())
            if 1 <= choice_num <= len(available_slots):
                selected_slot = available_slots[choice_num - 1]
        except ValueError:
            for slot in available_slots:
                if slot["time"].lower() in slot_choice.lower():
                    selected_slot = slot
                    break
        
        if not selected_slot:
            return "Could not detect the selected slot, please repeat."

        # Mark slot unavailable
        for slot in self.calendar_data["available_slots"]:
            if slot["id"] == selected_slot["id"]:
                slot["available"] = False
                break
        
        with open("mock_calendar.json", "w") as f:
            json.dump(self.calendar_data, f, indent=2)

        return f"Great! Booking confirmed for {selected_slot['date']} at {selected_slot['time']}."

    @function_tool
    async def search_faq(self, context: RunContext, query: str) -> str:
        query_lower = query.lower()
        
        for faq_item in self.company_data.get("faq", []):
            question = faq_item["question"].lower()
            if any(word in question for word in query_lower.split()):
                return faq_item["answer"]

        return "I don't have exact info on that—may I connect you with our team?"

    @function_tool
    async def store_lead_info(self, context: RunContext, field: str, value: str) -> str:
        self.lead_data[field] = value
        if field == "role":
            await self.detect_persona(context, value)
        return f"Got it, saved your {field}."

    @function_tool
    async def end_conversation(self, context: RunContext):
        return "Thank you! Have a great day."


# ============================================================
#  FRAUD ALERT AGENT (DAY 6)
# ============================================================

from fraud_repository import FraudRepository, FraudCase


class FraudAlertAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a calm, professional fraud detection representative for NovaBank.\n"
                "You are contacting the customer about a suspicious transaction.\n"
                "Follow this call flow:\n"
                "1. Greet politely.\n"
                "2. Ask for the user's first name.\n"
                "3. Call load_fraud_case_for_user(name).\n"
                "4. If no case → explain and end the call.\n"
                "5. Ask the stored security question.\n"
                "6. Verify using check_security_answer.\n"
                "7. If wrong twice → mark_verification_failed.\n"
                "8. If correct → read transaction details.\n"
                "9. Ask: 'Did you make this transaction?'\n"
                "10. If yes → mark_transaction_safe.\n"
                "11. If no → mark_transaction_fraudulent.\n"
                "12. End politely.\n"
            )
        )
        self.repo = FraudRepository()
        self.current_case: Optional[FraudCase] = None
        self.failed_attempts: int = 0

    @function_tool
    async def load_fraud_case_for_user(self, context: RunContext, user_name: str):
        case = self.repo.get_pending_case_for_user(user_name)
        if not case:
            self.current_case = None
            return {}
        self.current_case = case
        self.failed_attempts = 0
        return case.to_public_dict()

    @function_tool
    async def check_security_answer(self, context: RunContext, user_answer: str):
        if not self.current_case:
            return {"verified": False, "remaining_attempts": 0}

        correct = self.repo.verify_security_answer(self.current_case.id, user_answer)
        if correct:
            return {"verified": True, "remaining_attempts": 2}

        self.failed_attempts += 1
        remaining = max(0, 2 - self.failed_attempts)
        return {"verified": False, "remaining_attempts": remaining}

    @function_tool
    async def mark_verification_failed(self, context: RunContext):
        if not self.current_case:
            return {"updated": False}
        self.repo.update_status(
            self.current_case.id,
            "verification_failed",
            "Security verification failed.",
        )
        return {"updated": True}

    @function_tool
    async def mark_transaction_safe(self, context: RunContext):
        if not self.current_case:
            return {"updated": False}
        self.repo.update_status(
            self.current_case.id,
            "confirmed_safe",
            "Customer confirmed the transaction.",
        )
        return {"updated": True}

    @function_tool
    async def mark_transaction_fraudulent(self, context: RunContext):
        if not self.current_case:
            return {"updated": False}
        self.repo.update_status(
            self.current_case.id,
            "confirmed_fraud",
            "Customer denied the transaction.",
        )
        return {"updated": True}


# ============================================================
#   AGENT CHOOSER  (BASED ON ENV VARIABLE)
# ============================================================

def _create_agent() -> Agent:
    mode = os.getenv("AGENT_MODE", "sdr").lower()
    if mode == "fraud":
        logger.info("🚨 Launching Fraud Alert Agent")
        return FraudAlertAgent()
    else:
        logger.info("💼 Launching SDR Assistant")
        return SimpleSDRAssistant()


# ============================================================
#  ENTRYPOINT
# ============================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    load_dotenv(".env.local", override=True)

    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-anusha",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage Summary: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=_create_agent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
