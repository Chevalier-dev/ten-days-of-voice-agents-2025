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

logger = logging.getLogger("agent")

load_dotenv(".env.local")


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
9. Then say: "Great! Feel free to ask any questions about Razorpay, or I can help you schedule a demo."

INFORMATION COLLECTION RULES:
- ALWAYS ask for Name, Email, Company, Role, Team Size, Timeline, and Need FIRST
- Ask ONE question at a time, wait for answer
- Use store_lead_info() to save each detail immediately
- After collecting role, use detect_persona() to identify their persona type
- Don't proceed to demos or questions until you have all basic info

AFTER COLLECTING INFO - FAQ PHASE:
- Answer their questions using search_faq() tool
- Use persona-specific language based on detected persona
- While answering questions, collect pain points and interests naturally
- ONLY when they say "no more questions", then offer demo

DEMO OFFERING (ONLY AFTER FAQ PHASE):
- When customer indicates no more questions, say: "Would you like to schedule a demo to see this in action?"
- If they say YES to demo: Proceed to booking process

BOOKING PROCESS (ONLY IF DEMO ACCEPTED):
1. Ask: "What are the main challenges you're facing with payments?"
2. Ask: "What specific features would you like to see in the demo?"
3. Then say: "Perfect! Let me show you available times."
4. ALWAYS call show_available_meetings() to check real slots
5. Present ONLY available options from the tool
6. Use book_meeting() with their choice
7. Confirm with their name and email

CRITICAL RULES:
- NEVER skip the opening sequence
- NEVER offer demo until customer says "no more questions"
- NEVER book without email
- ALWAYS store info using store_lead_info() immediately
- Keep responses SHORT (1-2 sentences max)
- Ask one question at a time""",
        )
    
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
    
    def _has_required_info(self) -> bool:
        """Check if all required information has been collected"""
        required_fields = ["name", "email", "company", "role", "team_size", "timeline", "use_case"]
        return all(self.lead_data.get(field) for field in required_fields)
    
    async def _save_lead_data(self) -> str:
        """Save lead data to JSON file immediately"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"leads/lead_{timestamp}.json"
        
        lead_summary = {
            "timestamp": datetime.now().isoformat(),
            "lead_data": self.lead_data,
            "conversation_transcript": self.conversation_transcript,
            "detected_persona": self.detected_persona
        }
        
        os.makedirs("leads", exist_ok=True)
        with open(filename, "w") as f:
            json.dump(lead_summary, f, indent=2)
        
        logger.info(f"💾 Lead data saved to {filename}")
        return filename

    @function_tool
    async def detect_persona(self, context: RunContext, user_input: str) -> str:
        """Detect user persona based on their language and role."""
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
                return f"Got it! As a {self.detected_persona}, I can share how Razorpay specifically helps people in your role."
        
        return "Thanks for sharing! Let me understand your specific needs better."
    
    @function_tool
    async def show_available_meetings(self, context: RunContext, meeting_type: str = "demo") -> str:
        """Show available meeting slots for scheduling."""
        available_slots = []
        
        for slot in self.calendar_data.get("available_slots", []):
            if slot.get("available", False) and slot.get("type") == meeting_type:
                available_slots.append(slot)
        
        if not available_slots:
            return "I don't see any available slots for that meeting type right now. Would you like to try a different time?"
        
        options = available_slots[:5]
        name = self.lead_data.get("name", "")
        greeting = f"Great {name}! " if name else ""
        response = f"{greeting}Here are my available times:\n\n"
        
        for i, slot in enumerate(options, 1):
            response += f"{i}. {slot['date']} at {slot['time']} ({slot['duration']})\n"
        
        response += "\nWhich slot works best for you? Just say the number."
        return response
    
    @function_tool
    async def book_meeting(self, context: RunContext, slot_choice: str, meeting_type: str = "demo") -> str:
        """Book a meeting slot. REQUIRES email to be collected first."""
        
        # Check if email is collected
        if not self.lead_data.get("email"):
            return "I need your email address first to send the meeting confirmation. What's your email?"
        
        available_slots = [s for s in self.calendar_data.get("available_slots", []) 
                          if s.get("available", False) and s.get("type") == meeting_type]
        
        if not available_slots:
            return "I don't see any available slots for that meeting type. Let me check other options."
        
        selected_slot = None
        
        try:
            choice_num = int(slot_choice.strip())
            if 1 <= choice_num <= len(available_slots):
                selected_slot = available_slots[choice_num - 1]
        except ValueError:
            for slot in available_slots:
                if slot["time"].lower() in slot_choice.lower() or slot["date"] in slot_choice:
                    selected_slot = slot
                    break
        
        if not selected_slot:
            return "I didn't catch which time you prefer. Could you say the number or specific time again?"
        
        meeting_details = {
            "id": f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "slot_id": selected_slot["id"],
            "date": selected_slot["date"],
            "time": selected_slot["time"],
            "duration": selected_slot["duration"],
            "type": meeting_type,
            "lead_name": self.lead_data.get("name", "Prospect"),
            "lead_email": self.lead_data.get("email", ""),
            "lead_company": self.lead_data.get("company", ""),
            "booked_at": datetime.now().isoformat()
        }
        
        # Mark slot as unavailable
        for slot in self.calendar_data["available_slots"]:
            if slot["id"] == selected_slot["id"]:
                slot["available"] = False
                break
        
        # Add to booked meetings
        self.calendar_data["booked_meetings"].append(meeting_details)
        
        # Save calendar changes
        with open("mock_calendar.json", "w") as f:
            json.dump(self.calendar_data, f, indent=2)
        
        # Store meeting in lead data
        self.lead_data["booked_meeting"] = meeting_details
        
        # Save lead data immediately after booking
        await self._save_lead_data()
        
        name = self.lead_data.get("name", "")
        email = self.lead_data.get("email", "")
        return f"✅ Perfect! Meeting booked for {selected_slot['date']} at {selected_slot['time']}. I'll send you a confirmation. Thanks {name}!"

    @function_tool
    async def search_faq(self, context: RunContext, query: str) -> str:
        """Search company FAQ for relevant information."""
        query_lower = query.lower()
        
        # Simple keyword matching
        for faq_item in self.company_data.get("faq", []):
            question = faq_item["question"].lower()
            answer = faq_item["answer"]
            
            # Check if query keywords match question
            if any(word in question for word in query_lower.split()):
                return answer
        
        # Check products if no FAQ match
        for product in self.company_data.get("products", []):
            if any(word in product["name"].lower() for word in query_lower.split()):
                return f"{product['name']}: {product['description']}"
        
        return "I don't have specific information about that. Let me connect you with our team for detailed information."
    
    @function_tool
    async def store_lead_info(self, context: RunContext, field: str, value: str) -> str:
        """Store lead information as it's collected during conversation."""
        
        # Handle list fields (pain_points, key_interests)
        if field in ["pain_points", "key_interests"]:
            if field not in self.lead_data:
                self.lead_data[field] = []
            # Add to list if not already there
            if isinstance(value, str):
                # Split by common separators and add each item
                items = [item.strip() for item in value.replace(" and ", ", ").split(",")]
                for item in items:
                    if item and item not in self.lead_data[field]:
                        self.lead_data[field].append(item)
            logger.info(f"Stored lead info: {field} = {self.lead_data[field]}")
            return f"Got it, I've noted that down."
        else:
            self.lead_data[field] = value
            logger.info(f"Stored lead info: {field} = {value}")
            
            # Auto-detect persona when role is stored
            if field == "role":
                await self.detect_persona(context, value)
            
            return f"Got it, I've noted your {field}."
    
    @function_tool
    async def end_conversation(self, context: RunContext) -> str:
        """End the conversation and save lead data."""
        if self.conversation_ended:
            return "Thank you for your time!"
        
        self.conversation_ended = True
        
        # Save final lead data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"leads/complete_lead_{timestamp}.json"
        
        lead_summary = {
            "timestamp": datetime.now().isoformat(),
            "lead_data": self.lead_data,
            "conversation_transcript": self.conversation_transcript,
            "detected_persona": self.detected_persona
        }
        
        os.makedirs("leads", exist_ok=True)
        with open(filename, "w") as f:
            json.dump(lead_summary, f, indent=2)
        
        logger.info(f"Complete lead data saved to {filename}")
        
        name = self.lead_data.get("name", "")
        company = self.lead_data.get("company", "your company")
        
        return f"Thanks {name}! I've saved all your information. Our team will follow up with {company} soon. Have a great day!"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    # Load environment variables
    load_dotenv(".env.local", override=True)
    
    # Get Gemini API key
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not gemini_api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        raise ValueError("GOOGLE_API_KEY is required")
    
    logger.info(f"Gemini API Key present: {bool(gemini_api_key)}")

    # Set up voice AI pipeline using Gemini, Murf, Deepgram
    session = AgentSession(
        # Speech-to-text (STT)
        stt=deepgram.STT(model="nova-3"),
        
        # Large Language Model (LLM) - Using Gemini instead of Azure
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        
        # Text-to-speech (TTS)
        tts=murf.TTS(
            voice="en-IN-anusha", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        
        # VAD and turn detection
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        
        # Allow preemptive generation
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=SimpleSDRAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))