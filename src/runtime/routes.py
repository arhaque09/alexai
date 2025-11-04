# api.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from fastapi.logger import logger

from main import (
    User,
    Personality,
    AiPersona,
    enforce_with_retry,
    maybe_update_persona_from_text,
    create_new_personality,
    model_opening_line,
    regenerate_with_critique,
    evolve_personality,
    sanitize_user_input,
    blend_rgb
)

# ---- Your adapter (HF / QLoRA) ----
from adapter import HFAdapterClient

# =============================================================================
# FastAPI App Setup
# =============================================================================

app = FastAPI(title="Alex Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# State / Session Management (in-memory)
# =============================================================================

class Session:
    def __init__(self, user: User):
        self.user: User = user
        self.ai: Optional[AiPersona] = None
        self.history: List[Dict[str, str]] = []
        self.scenario: Optional[str] = None
        self.choice_payload: Optional[Dict[str, Any]] = None

# Global, in-memory session store keyed by user_id (e.g., MongoDB _id)
SESSIONS: Dict[str, Session] = {}

# Shared LLM instance (created once via /setup)
LLM: Optional[HFAdapterClient] = None


# =============================================================================
# Pydantic Schemas
# =============================================================================

class UserCreate(BaseModel):
    user_id: str = Field(..., description="Stable id for the user (e.g. Mongo _id)")
    name: str
    age: int

class ChoiceIn(BaseModel):
    user_id: str
    tone: str = Field(..., description="One of: 'random', a tone like 'warm', or 'Create new Personality'")
    traits: Optional[str] = None
    sarcasm: Optional[str] = None
    formality: Optional[str] = None
    emoji_usage: Optional[str] = None
    tone_blend: Optional[Dict[str, float]] = None
    amount_of_slang: Optional[str] = None
    type_of_slang: Optional[str] = None
    wordiness: Optional[str] = None
    childhood: Optional[str] = None               # comma+space list as in main
    childhood_adjectives: Optional[str] = None    # comma+space list as in main
    special_instructions: Optional[str] = None
    likes: Optional[str] = None                   # comma+space list
    dislikes: Optional[str] = None                # comma+space list
    hobbies: Optional[str] = None                 # comma+space list

class SetupIn(BaseModel):
    user_id: str

class LoopChoiceIn(BaseModel):
    user_id: str
    scenario: str

class UserMessageIn(BaseModel):
    user_id: str
    message: str
    regenerate_critique: Optional[str] = Field(
        None, description="If provided, revises last assistant reply using this critique"
    )

class SimpleOk(BaseModel):
    ok: bool = True

class OpeningLineOut(BaseModel):
    opening: str

class AIReplyOut(BaseModel):
    reply: str
    persona_tone: str
    tone_blend: Dict[str, float]
    history_len: int

class Color(BaseModel): 
    red: str
    blue: str
    green: str
class UserInfoOut(BaseModel):
    user_id: str
    name: str
    age: int
    is_minor: bool

class PersonaSnapshot(BaseModel):
    tone: str
    tone_blend: Dict[str, float]
    traits: List[str]
    formality: str
    sarcasm: str
    emoji_usage: str
    wordiness: str
    likes: List[str]
    dislikes: List[str]
    hobbies: List[str]
    childhood: List[str]
    childhood_adjectives: List[str]

class SetupOut(BaseModel):
    persona: PersonaSnapshot


# =============================================================================
# Helpers
# =============================================================================

def get_session_or_404(user_id: str) -> Session:
    sess = SESSIONS.get(user_id)
    if not sess:
        logger.error(f"Session not found for user_id: {user_id}")  # Log missing session
        raise HTTPException(status_code=404, detail="Session not found for user_id")
    logger.info(f"Session retrieved for user_id: {user_id}")  # Log session retrieval
    return sess

def persona_snapshot(p: Personality) -> PersonaSnapshot:
    return PersonaSnapshot(
        tone=p.tone,
        tone_blend=p.tone_blend or {p.tone: 1.0},
        traits=list(getattr(p, "traits", []) or []),
        formality=p.formality,
        sarcasm=p.sarcasm,
        emoji_usage=p.emoji_usage,
        wordiness=p.wordiness,
        likes=list(getattr(p, "likes", []) or []),
        dislikes=list(getattr(p, "dislikes", []) or []),
        hobbies=list(getattr(p, "hobbies", []) or []),
        childhood=list(getattr(p, "childhood", []) or []),
        childhood_adjectives=list(getattr(p, "childhood_adjectives", []) or []),
    )

def ensure_llm_initialized() -> HFAdapterClient:
    global LLM
    if LLM is None:
        # Mirror your __main__ adapter initialization
        PROJECT_ROOT = Path(__file__).resolve().parents[0]  # adjust if repo layout differs
        ADAPTER_DIR = PROJECT_ROOT / "outputs" / "alex-qlora"
        # If your original code uses parents[2], keep that:
        if not ADAPTER_DIR.exists():
            PROJECT_ROOT = Path(__file__).resolve().parents[2]
            ADAPTER_DIR = PROJECT_ROOT / "outputs" / "alex-qlora"

        LLM = HFAdapterClient(
            base_model="Qwen/Qwen2.5-3B-Instruct",
            adapter_dir=str(ADAPTER_DIR),
            use_4bit=False,
        )
    return LLM


# =============================================================================
# Routes
# =============================================================================

@app.get("/health", response_model=SimpleOk)
def health() -> SimpleOk:
    return SimpleOk()


# ----- User / Session --------------------------------------------------------

@app.post("/user", response_model=SimpleOk)
def create_or_update_user(payload: UserCreate) -> SimpleOk:
    if payload.age < 0:
        raise HTTPException(status_code=400, detail="Age must be non-negative")

    user = User(payload.name, payload.age)
    setattr(user, "is_minor", payload.age < 18)

    SESSIONS[payload.user_id] = Session(user=user)
    logger.info(f"Session created for user_id: {payload.user_id}")  # Log session creation
    return SimpleOk()


@app.get("/user_info", response_model=UserInfoOut)
def get_user_info(user_id: str = Query(...)) -> UserInfoOut:
    sess = get_session_or_404(user_id)
    user = sess.user
    return UserInfoOut(
        user_id=user_id,
        name=user.name,
        age=user.age,
        is_minor=getattr(user, "is_minor", False),
    )


# ----- Choice / Setup --------------------------------------------------------

@app.post("/choice", response_model=SimpleOk)
def set_choice(choice: ChoiceIn) -> SimpleOk:
    if not choice.tone:
        logger.error("Tone is missing in the /choice payload")
        raise HTTPException(status_code=422, detail="Tone is required in the payload.")

    sess = get_session_or_404(choice.user_id)
    sess.choice_payload = choice.dict()
    logger.info(f"Choice set for user_id: {choice.user_id}, tone: {choice.tone}")
    return SimpleOk()


@app.post("/setup", response_model=SetupOut)
def setup(payload: SetupIn) -> SetupOut:
    sess = get_session_or_404(payload.user_id)
    user = sess.user

    # Adult-only gate (mirrors __main__)
    if getattr(user, "is_minor", False):
        raise HTTPException(status_code=403, detail="Adults (18+) only.")

    if not sess.choice_payload:
        raise HTTPException(status_code=400, detail="No /choice posted yet for this user.")

    llm = ensure_llm_initialized()

    choice = sess.choice_payload
    tone = (choice.get("tone") or "").strip()

    # Create a baseline AiPersona (random or not)
    is_random = tone.lower() == "random"
    ai = AiPersona(user, is_random=is_random)
    ai.createPersonality()

    if not is_random:
        if tone == "Create new Personality":
            # Build a fully custom personality
            pers = create_new_personality(
                user=user,
                traits=(choice.get("traits") or ""),
                sarcasm=(choice.get("sarcasm") or "low"),
                formality=(choice.get("formality") or "medium"),
                emoji_usage=(choice.get("emoji_usage") or "low"),
                tone_blend=(choice.get("tone_blend") or {"neutral": 1.0}),
                amount_of_slang=(choice.get("amount_of_slang") or "low"),
                type_of_slang=(choice.get("type_of_slang") or "genx"),
                wordiness=(choice.get("wordiness") or "medium"),
                childhood_adjectives=(choice.get("childhood_adjectives") or ""),
                childhood_mems=(choice.get("childhood") or ""),
                special_instructions=(choice.get("special_instructions") or ""),
                likes=(choice.get("likes") or ""),
                dislikes=(choice.get("dislikes") or ""),
                hobbies=(choice.get("hobbies") or ""),
            )
            ai.personality = pers
            # ensure dominant tone set from tone_blend
            ai.personality.tone = max(ai.personality.tone_blend, key=ai.personality.tone_blend.get)
        else:
            # They picked a specific tone from the menu
            ai.personality.tone = tone
            ai.personality.setPersonality()

    sess.ai = ai
    return SetupOut(persona=persona_snapshot(ai.personality))


# ----- Scenario / Opening ----------------------------------------------------

#gives adds scenario
@app.post("/loop_choice", response_model=SimpleOk)
def set_loop_choice(payload: LoopChoiceIn) -> SimpleOk:
    sess = get_session_or_404(payload.user_id)
    sess.scenario = payload.scenario or "random"
    
    return SimpleOk()

@app.get("/tone_color", response_model=Color)
def find_color(user_id: str = Query(...)):
    sess = get_session_or_404(user_id)
    if not sess.ai:
        logger.error(f"AI not initialized for user_id: {user_id}")
        raise HTTPException(status_code=400, detail="Call /setup first to initialize AI.")

    ai = sess.ai
    TONE_COLORS = {
        "random":       (255, 255, 255),  # white
        "loving":       (255, 105, 180),  # hot pink
        "harsh":        (255, 59, 48),    # red
        "warm":         (255, 165, 66),   # orange
        "playful":      (102, 204, 255),  # light blue
        "formal":       (30, 58, 138),    # dark blue
        "casual":       (72, 187, 120),   # teal/green
        "authoritative": (88, 86, 214),   # indigo
        "cautious":     (255, 204, 0),    # amber
        "direct":       (0, 188, 212),    # cyan-ish
        "diplomatic":   (76, 175, 80),    # green
        "neutral":      (160, 160, 160),  # gray
        "curious":      (156, 39, 176),   # purple
        "witty_dry":    (112, 128, 144),  # slate
        "reflective":   (100, 92, 170),   # violet
    }
    r, b, g = blend_rgb(ai.personality.tone_blend, TONE_COLORS)
    return Color(
        red=str(r),
        blue=str(b),
        green=str(g)
    )
    
@app.get("/opening_line", response_model=OpeningLineOut)
def opening_line(user_id: str = Query(...)) -> OpeningLineOut:
    sess = get_session_or_404(user_id)
    if not sess.ai:
        logger.error(f"AI not initialized for user_id: {user_id}")
        raise HTTPException(status_code=400, detail="Call /setup first to initialize AI.")

    user = sess.user
    ai = sess.ai
    llm = ensure_llm_initialized()

    scenario = (sess.scenario or "random").strip().lower()

    if scenario == "random":
        opening = model_opening_line(llm, ai.personality, user, True)
    else:
        opening = model_opening_line(llm, ai.personality, user, False, sess.scenario)

    # Save assistant message to history
    sess.history.append({"role": "assistant", "content": opening})
    return OpeningLineOut(opening=opening)


# ----- Chat Loop / Messages --------------------------------------------------

@app.post("/user_response", response_model=AIReplyOut)
def user_response(body: UserMessageIn) -> AIReplyOut:
    sess = get_session_or_404(body.user_id)
    if not sess.ai:
        raise HTTPException(status_code=400, detail="Call /setup first.")

    user = sess.user
    if getattr(user, "is_minor", False):
        raise HTTPException(status_code=403, detail="Adults (18+) only.")

    ai = sess.ai
    llm = ensure_llm_initialized()

    user_msg = sanitize_user_input(body.message.strip())
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message.")

    # If a critique is provided and we have at least one assistant reply, run regenerate
    if body.regenerate_critique:
        has_assistant = any(m.get("role") == "assistant" for m in sess.history)
        if not has_assistant:
            raise HTTPException(status_code=400, detail="No assistant reply to regenerate.")
        reply = regenerate_with_critique(sess.history, body.regenerate_critique, llm, ai.personality)
        # append regeneration as a new assistant reply
        sess.history.append({"role": "user", "content": f"REGENERATE: {body.regenerate_critique}"})
        sess.history.append({"role": "assistant", "content": reply})
    else:
        # Normal turn
        maybe_update_persona_from_text(user_msg, ai, llm, sess.history)
        reply = enforce_with_retry(llm, ai.personality, sess.history, user_msg)
        sess.history.append({"role": "user", "content": user_msg})
        sess.history.append({"role": "assistant", "content": reply})

        # Evolve persona every 3 user turns (mirrors __main__)
        user_turns = sum(1 for m in sess.history if m["role"] == "user")
        if user_turns > 0 and user_turns % 3 == 0:
            evolve_personality(ai.personality, sess.history, llm)

    return AIReplyOut(
        reply=reply,
        persona_tone=ai.personality.tone,
        tone_blend=ai.personality.tone_blend or {ai.personality.tone: 1.0},
        history_len=len(sess.history),
    )


@app.get("/ai_response", response_model=AIReplyOut)
def get_last_ai_response(user_id: str = Query(...)) -> AIReplyOut:
    sess = get_session_or_404(user_id)
    if not sess.ai:
        raise HTTPException(status_code=400, detail="Call /setup first.")
    # find last assistant message
    last = None
    for m in reversed(sess.history):
        if m.get("role") == "assistant":
            last = m.get("content")
            break
    if not last:
        raise HTTPException(status_code=404, detail="No assistant reply found yet.")

    ai = sess.ai
    return AIReplyOut(
        reply=last,
        persona_tone=ai.personality.tone,
        tone_blend=ai.personality.tone_blend or {ai.personality.tone: 1.0},
        history_len=len(sess.history),
    )


# =============================================================================
# Local Dev
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
