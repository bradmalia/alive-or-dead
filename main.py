"""
Alive or Dead — FastAPI backend.

Structure:
  1. Imports & logging
  2. Environment / configuration constants
  3. Pydantic models & dataclasses
  4. HTML parser for fragment validation
  5. Module-level state (sessions, caches, AI client)
  6. Utility functions (perf logging, audit log, JSON extraction, …)
  7. Wikimedia helpers (portrait search, status verification)
  8. AI generation pipeline (prompt building, SDK wrappers, round normalisation)
  9. Session orchestration (prefetch, round queue)
 10. FastAPI app, middleware, and HTTP route handlers
"""
import asyncio
import collections
import concurrent.futures
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
from html.parser import HTMLParser
import json
import logging
import os
import re
import random
import threading
import time
import unicodedata
import uuid
import sqlite3
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request as UrlRequest, urlopen
import socket

# Patch socket.getaddrinfo to force IPv4 and prevent AWS IPv6 blackhole timeouts
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from pydantic import field_validator

try:
    from google import genai as modern_genai
    from google.genai import types as modern_types
except ImportError:
    modern_genai = None
    modern_types = None

try:
    import google.generativeai as legacy_genai
except ImportError:
    legacy_genai = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DB_PATH = Path("leaderboard.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            initials TEXT NOT NULL,
            score INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if legacy_genai and api_key:
    legacy_genai.configure(api_key=api_key)

BASE_DIR = Path(__file__).resolve().parent
GLOBAL_HISTORY_FILE = BASE_DIR / "global_history.json"


def normalize_app_base_path(raw_value: str | None) -> str:
    value = (raw_value or "").strip()
    if not value or value == "/":
        return ""
    if not value.startswith("/"):
        value = "/" + value
    return value.rstrip("/")


def app_path(path: str) -> str:
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{APP_BASE_PATH}{normalized_path}" if APP_BASE_PATH else normalized_path

APP_VERSION = "3.6.0"
APP_BASE_PATH = normalize_app_base_path(os.getenv("APP_BASE_PATH", ""))
# CORS: set CORS_ALLOWED_ORIGINS to a comma-separated list of origins to restrict
# cross-origin access (e.g. "https://example.com,https://www.example.com").
# Leave unset to allow all origins (no credentials forwarded either way).
_CORS_ALLOWED_ORIGINS_RAW = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
CORS_ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _CORS_ALLOWED_ORIGINS_RAW.split(",") if o.strip()]
    if _CORS_ALLOWED_ORIGINS_RAW
    else ["*"]
)
ROUNDS_PER_GAME = 10
GLOBAL_HISTORY_PROMPT_LIMIT = int(os.getenv("GLOBAL_HISTORY_PROMPT_LIMIT", "50"))
ROUND_GENERATION_ATTEMPTS = int(os.getenv("ROUND_GENERATION_ATTEMPTS", "3"))
PREFETCH_ROUND_BUFFER = int(os.getenv("PREFETCH_ROUND_BUFFER", "2"))
CANDIDATE_SELECTION_COUNT = min(
    50,
    max(1, int(os.getenv("CANDIDATE_SELECTION_COUNT", "30"))),
)
LOCAL_CANDIDATE_FIRST_HISTORY_SIZE = int(
    os.getenv("LOCAL_CANDIDATE_FIRST_HISTORY_SIZE", "250")
)
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.35"))
GENERATION_TOP_P = float(os.getenv("GENERATION_TOP_P", "0.8"))
GENERATION_MAX_OUTPUT_TOKENS = int(os.getenv("GENERATION_MAX_OUTPUT_TOKENS", "8192"))
CANDIDATE_SELECTION_MAX_OUTPUT_TOKENS = int(
    os.getenv("CANDIDATE_SELECTION_MAX_OUTPUT_TOKENS", "2000")
)
RACE_FAST_MODELS = os.getenv("RACE_FAST_MODELS", "false").lower() == "true"
# Session lifetime: sessions older than this are automatically evicted.
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 90)))       # default 90 min
SESSION_CLEANUP_INTERVAL_SECONDS = int(os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", "300"))  # default 5 min
PERF_LOG_ENABLED = os.getenv("PERF_LOG_ENABLED", "true").lower() == "true"
GEMINI_AUDIT_LOG_ENABLED = os.getenv("GEMINI_AUDIT_LOG_ENABLED", "true").lower() == "true"
IMAGE_URL_VALIDATION_ENABLED = os.getenv("IMAGE_URL_VALIDATION_ENABLED", "true").lower() == "true"
STATUS_VERIFICATION_ENABLED = os.getenv("STATUS_VERIFICATION_ENABLED", "true").lower() == "true"
STATUS_VERIFICATION_REQUIRED = os.getenv("STATUS_VERIFICATION_REQUIRED", "true").lower() == "true"
IMAGE_URL_VALIDATION_TIMEOUT_SECONDS = float(
    os.getenv("IMAGE_URL_VALIDATION_TIMEOUT_SECONDS", "4.0")
)
WIKIMEDIA_SEARCH_TIMEOUT_SECONDS = float(
    os.getenv("WIKIMEDIA_SEARCH_TIMEOUT_SECONDS", "3.0")
)
WIKIMEDIA_SEARCH_LIMIT = int(os.getenv("WIKIMEDIA_SEARCH_LIMIT", "8"))
IMAGE_FETCH_USER_AGENT = os.getenv(
    "IMAGE_FETCH_USER_AGENT",
    f"AliveOrDeadPOC/{APP_VERSION} (image validation)",
)
WIKIMEDIA_API_USER_AGENT = os.getenv(
    "WIKIMEDIA_API_USER_AGENT",
    f"AliveOrDeadPOC-{uuid.uuid4().hex[:8]}/{APP_VERSION} (aliveordeadbot@example.com)",
)
FAST_MODEL_CANDIDATES = [
    item.strip()
    for item in os.getenv(
        "GEMINI_FAST_MODELS",
        "gemini-2.5-flash-lite",
    ).split(",")
    if item.strip()
]
BACKGROUND_MODEL_CANDIDATES = [
    item.strip()
    for item in os.getenv(
        "GEMINI_BACKGROUND_MODELS",
        "gemini-2.5-flash-lite",
    ).split(",")
    if item.strip()
]
GEMINI_AUDIT_LOG_FILE = Path(
    os.getenv("GEMINI_AUDIT_LOG_FILE", str(BASE_DIR / "gemini_audit.jsonl"))
).expanduser()
if not GEMINI_AUDIT_LOG_FILE.is_absolute():
    GEMINI_AUDIT_LOG_FILE = (BASE_DIR / GEMINI_AUDIT_LOG_FILE).resolve()
# Audit log rotation: rotate when file exceeds this size; keep this many backups.
GEMINI_AUDIT_LOG_MAX_BYTES = int(os.getenv("GEMINI_AUDIT_LOG_MAX_BYTES", str(50 * 1024 * 1024)))  # 50 MB
GEMINI_AUDIT_LOG_BACKUP_COUNT = int(os.getenv("GEMINI_AUDIT_LOG_BACKUP_COUNT", "3"))

GUESS_ALIVE_RE = re.compile(r"submitGuess\((['\"])alive\1\)")
GUESS_DEAD_RE = re.compile(r"submitGuess\((['\"])dead\1\)")
NEXT_ROUND_RE = re.compile(r"loadNextRound\(\)")
IMG_SRC_RE = re.compile(r"<img\b[^>]*\bsrc\s*=\s*(['\"])(.*?)\1", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
QUOTED_TEXT_RE = re.compile(r'"[^"]{1,160}"|\'[^\']{1,160}\'|“[^”]{1,160}”|‘[^’]{1,160}’')
REMOTE_IMAGE_PREFIXES = ("http://", "https://", "//")
WIKIMEDIA_DOMAIN_MARKERS = ("wikimedia.org", "wikipedia.org")
PORTRAIT_IMAGE_PLACEHOLDER = "__PORTRAIT_IMAGE_URL__"
WIKIMEDIA_COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
WIKIPEDIA_PAGE_SEARCH_API_URL = "https://api.wikimedia.org/core/v1/wikipedia/en/search/page"
WIKIPEDIA_PAGE_SUMMARY_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKIDATA_ENTITY_DATA_URL_TEMPLATE = "https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
WIKIDATA_INSTANCE_OF_PROPERTY = "P31"
WIKIDATA_DATE_OF_DEATH_PROPERTY = "P570"
WIKIDATA_HUMAN_ENTITY_ID = "Q5"
WIKIMEDIA_THUMB_PREFIXES = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/",
    "http://upload.wikimedia.org/wikipedia/commons/thumb/",
    "//upload.wikimedia.org/wikipedia/commons/thumb/",
)
VALID_WIKIMEDIA_THUMB_WIDTHS = {
    "20",
    "40",
    "60",
    "120",
    "250",
    "320",
    "330",
    "500",
    "640",
    "800",
    "960",
    "1280",
    "1920",
    "3840",
}
SUPPORTED_RASTER_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
COMMONS_QUERY_NOISE_RE = re.compile(r"\b(?:wikimedia|wikipedia|commons)\b", re.IGNORECASE)
COMMONS_CANDIDATE_BONUSES = (
    ("portrait", 6),
    ("headshot", 5),
    ("cropped", 2),
    ("publicity", 2),
    ("press", 2),
)
COMMONS_CANDIDATE_PENALTIES = (
    ("artwork", -10),
    ("caricature", -10),
    ("cartoon", -10),
    ("drawing", -10),
    ("illustration", -10),
    ("mural", -10),
    ("painting", -10),
    ("poster", -8),
    ("sculpture", -10),
    ("sketch", -10),
    ("statue", -10),
    ("tableau", -10),
    ("tape", -6),
    ("wax", -8),
    ("signature", -6),
    ("autograph", -6),
    ("logo", -6),
    ("performing", -2),
    ("concert", -2),
)
NON_PHOTO_TITLE_KEYWORDS = {
    "artwork",
    "caricature",
    "cartoon",
    "drawing",
    "illustration",
    "mural",
    "painting",
    "poster",
    "sculpture",
    "sketch",
    "statue",
    "tableau",
    "tape",
    "wax",
}
NAME_TOKEN_STOPWORDS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "sir",
    "dame",
    "lord",
    "lady",
    "jr",
    "sr",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "the",
}
NAME_TOKEN_PARTICLES = {
    "al",
    "bin",
    "da",
    "das",
    "de",
    "del",
    "della",
    "der",
    "di",
    "dos",
    "du",
    "el",
    "ibn",
    "la",
    "le",
    "st",
    "saint",
    "ten",
    "ter",
    "van",
    "von",
}
EMERGENCY_FALLBACK_NAMES = [
    "Harrison Ford",
    "Dolly Parton",
    "Susan Sarandon",
    "Jane Fonda",
    "Bill Murray",
    "Stevie Nicks",
    "Martin Scorsese",
    "Cher",
    "Elton John",
    "Dick Van Dyke",
    "Tom Hanks",
    "Angela Lansbury",
    "Marilyn Monroe",
    "Freddie Mercury",
    "Betty White",
    "Joan Rivers",
    "David Bowie",
    "Robin Williams",
    "Carrie Fisher",
    "Lucille Ball",
    "Muhammad Ali",
    "Paul Newman",
    "Aretha Franklin",
    "Marlon Brando",
    "Julie Christie",
    "Vanessa Redgrave",
    "Faye Dunaway",
    "Elliott Gould",
    "Sally Field",
    "Lily Tomlin",
    "Bette Midler",
    "Goldie Hawn",
    "Kurt Russell",
    "Michael Douglas",
    "Kathleen Turner",
    "Geena Davis",
    "Candice Bergen",
    "Glenn Close",
    "Jessica Lange",
    "Michelle Pfeiffer",
    "Jamie Lee Curtis",
    "Danny DeVito",
    "Michael Keaton",
    "Kevin Bacon",
    "John Lithgow",
    "Bryan Cranston",
    "Diana Ross",
    "Carole King",
    "Dionne Warwick",
    "Smokey Robinson",
    "Gladys Knight",
    "Patti LaBelle",
    "Bonnie Raitt",
    "James Taylor",
    "Iggy Pop",
    "Alice Cooper",
    "Rod Stewart",
    "Sting",
    "Phil Collins",
    "Peter Gabriel",
    "Annie Lennox",
    "Cyndi Lauper",
    "Boy George",
    "Billy Joel",
    "Wayne Gretzky",
    "Martina Navratilova",
    "Billie Jean King",
    "Nadia Comaneci",
    "Carl Lewis",
    "Mark Spitz",
    "Jack Nicklaus",
    "Gary Player",
    "John McEnroe",
    "Chris Evert",
    "Jane Goodall",
    "Margaret Atwood",
    "Salman Rushdie",
    "Isabel Allende",
    "Andrew Lloyd Webber",
    "Arthur Miller",
    "Tennessee Williams",
    "Marlene Dietrich",
    "Bette Davis",
    "Greta Garbo",
    "Katharine Hepburn",
    "Ava Gardner",
    "Rita Hayworth",
    "Peter O'Toole",
    "Omar Sharif",
    "Halle Berry",
    "Sandra Oh",
    "Lucy Liu",
    "Ming-Na Wen",
    "Michelle Yeoh",
    "Chow Yun-fat",
    "Jet Li",
    "Gong Li",
    "Pedro Almodovar",
    "Werner Herzog",
    "Ken Burns",
    "Spike Lee",
    "Ridley Scott",
    "Martin Sheen",
    "Allison Janney",
    "Frances McDormand",
    "Meryl Streep",
    "Viola Davis",
    "Laura Dern",
    "Laura Linney",
    "Diane Lane",
    "Angela Bassett",
    "Joni Mitchell",
    "Jeff Bridges",
    "Cary Grant",
    "Ingrid Bergman",
    "Janis Joplin",
    "Amy Winehouse",
]
BUTTON_OVERLAY_CLASS_PREFIXES = (
    "inset-",
    "-inset-",
    "top-",
    "-top-",
    "right-",
    "-right-",
    "bottom-",
    "-bottom-",
    "left-",
    "-left-",
    "translate-",
    "-translate-",
    "-m",
)
BUTTON_OVERLAY_CLASS_EXACT = {
    "absolute",
    "fixed",
    "sticky",
}
PROHIBITED_FRAGMENT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"<\s*script\b",
        r"<\s*style\b",
        r"<\s*html\b",
        r"<\s*body\b",
        r"<\s*head\b",
        r"data:image",
        r"base64",
        # Block javascript: pseudo-protocol in href/src/action attributes.
        r"""javascript\s*:""",
        # NOTE: inline on* event-handler attribute validation is performed by
        # validate_fragment_event_handlers(), which allowlists the two known-safe
        # game callbacks (submitGuess / loadNextRound) while rejecting everything else.
    ]
]
# Allowlisted onclick values for AI-generated fragments.  Any onclick whose
# normalised value does not match one of these is rejected as unsafe.
_SAFE_ONCLICK_RE = re.compile(
    r"""^\s*(?:submitGuess\(\s*['"](?:alive|dead)['"]\s*\)|loadNextRound\(\s*\))\s*$""",
    re.IGNORECASE,
)
# Any HTML attribute name that begins with "on" is an event handler.
_EVENT_ATTR_RE = re.compile(r"^on\w+$", re.IGNORECASE)
UNSUPPORTED_TAILWIND_CLASS_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\baspect-w-\d+\b",
        r"\baspect-h-\d+\b",
    ]
]
SPOILER_HINT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bassassinat(?:ed|ion)?\b",
        r"\bmurder(?:ed|er)?\b",
        r"\bkilled\b",
        r"\bdied\b",
        r"\bdeath\b",
        r"\bfuneral\b",
        r"\bburied\b",
        r"\bgrave\b",
        r"\bmemorial\b",
        r"\bposthum",
        r"\bthe late\b",
        r"\btribute\b",
        r"\bstill alive\b",
        r"\bpassed away\b",
        r"\bdeceased\b",
    ]
]


class GeneratedRound(BaseModel):
    person_name: str = Field(min_length=1, max_length=100)
    actual_status: Literal["alive", "dead"]
    portrait_search_query: str = Field(min_length=3, max_length=200)
    guessing_ui_html: str = Field(min_length=50, max_length=5000)
    reveal_ui_html: str = Field(min_length=50, max_length=5000)


class CandidatePerson(BaseModel):
    person_name: str = Field(min_length=1, max_length=100)
    actual_status: Literal["alive", "dead"] | None = None

    @field_validator("actual_status", mode="before")
    @classmethod
    def normalize_actual_status(cls, value):
        if value is None:
            return value
        if isinstance(value, str):
            return value.strip().lower()
        return value


class CandidateSelection(BaseModel):
    candidates: list[CandidatePerson] = Field(min_length=1, max_length=50)


@dataclass(frozen=True)
class LockedCandidate:
    person_name: str
    actual_status: str | None = None
    source: str = "candidate_selection"


@dataclass
class FragmentNode:
    tag: str
    attrs: dict[str, str]
    parent: "FragmentNode | None" = None
    children: list["FragmentNode"] = field(default_factory=list)


class FragmentTreeParser(HTMLParser):
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = FragmentNode(tag="_root", attrs={})
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = FragmentNode(
            tag=tag.lower(),
            attrs={key.lower(): (value or "") for key, value in attrs},
            parent=self.stack[-1],
        )
        self.stack[-1].children.append(node)
        if node.tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == lowered:
                del self.stack[index:]
                break

sessions: dict[str, dict] = {}
runtime_state = {
    "last_error": None,
    "last_model": None,
}
modern_client = modern_genai.Client(api_key=api_key) if modern_genai and api_key else None
gemini_audit_log_lock = threading.Lock()
global_history_lock = threading.Lock()   # guards global_history.json read-modify-write


class BoundedCache:
    """Thread-safe LRU cache with a configurable maximum entry count.

    Uses :class:`collections.OrderedDict` to track insertion/access order.
    The least-recently-used entry is evicted when ``maxsize`` is exceeded.
    Replaces the plain ``dict`` + separate ``threading.Lock`` pattern used
    previously, which grew without bound.
    """

    def __init__(self, maxsize: int = 2000) -> None:
        self._data: collections.OrderedDict = collections.OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def set(self, key, value) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "2000"))

image_url_validation_cache: BoundedCache = BoundedCache(CACHE_MAX_SIZE)
portrait_resolution_cache: BoundedCache = BoundedCache(CACHE_MAX_SIZE)
wikipedia_person_summary_cache: BoundedCache = BoundedCache(CACHE_MAX_SIZE)
wikidata_entity_cache: BoundedCache = BoundedCache(CACHE_MAX_SIZE)
status_verification_cache: BoundedCache = BoundedCache(CACHE_MAX_SIZE)


def perf_now() -> float:
    return time.perf_counter()


def log_perf(stage: str, started_at: float, **fields) -> float:
    duration_ms = round((time.perf_counter() - started_at) * 1000, 1)
    if PERF_LOG_ENABLED:
        payload = {"stage": stage, "duration_ms": duration_ms, **fields}
        logger.info("perf %s", json.dumps(payload, sort_keys=True, default=str))
    return duration_ms


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rotate_audit_log_if_needed() -> None:
    """Rotate the audit log if it exceeds GEMINI_AUDIT_LOG_MAX_BYTES.
    Caller must already hold gemini_audit_log_lock.
    Produces up to GEMINI_AUDIT_LOG_BACKUP_COUNT numbered backup files
    (e.g. gemini_audit.1.jsonl, gemini_audit.2.jsonl, …).
    """
    if GEMINI_AUDIT_LOG_MAX_BYTES <= 0:
        return
    try:
        size = GEMINI_AUDIT_LOG_FILE.stat().st_size if GEMINI_AUDIT_LOG_FILE.exists() else 0
    except OSError:
        return
    if size < GEMINI_AUDIT_LOG_MAX_BYTES:
        return
    # Shift existing backups: .2 → .3, .1 → .2
    for i in range(GEMINI_AUDIT_LOG_BACKUP_COUNT - 1, 0, -1):
        src = GEMINI_AUDIT_LOG_FILE.with_suffix(f".{i}.jsonl")
        dst = GEMINI_AUDIT_LOG_FILE.with_suffix(f".{i + 1}.jsonl")
        try:
            if src.exists():
                src.replace(dst)
        except OSError as exc:
            logger.warning("Audit log rotation shift failed (%s → %s): %s", src.name, dst.name, exc)
    # Move the active log to .1
    try:
        GEMINI_AUDIT_LOG_FILE.replace(GEMINI_AUDIT_LOG_FILE.with_suffix(".1.jsonl"))
        logger.info("Audit log rotated: %s", GEMINI_AUDIT_LOG_FILE.name)
    except OSError as exc:
        logger.warning("Audit log rotation failed: %s", exc)


def append_gemini_audit_log(event_type: str, **fields) -> None:
    if not GEMINI_AUDIT_LOG_ENABLED:
        return

    payload = {
        "timestamp_utc": utc_now_iso(),
        "event": event_type,
        **fields,
    }

    try:
        GEMINI_AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        with gemini_audit_log_lock:
            _rotate_audit_log_if_needed()
            with GEMINI_AUDIT_LOG_FILE.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:
        logger.warning("Failed to write Gemini audit log: %s", exc)


SYSTEM_PROMPT = """
Generate exactly one round for an "Alive or Dead" guessing game.

Return only raw JSON with:
- `person_name`
- `actual_status`
- `portrait_search_query`
- `guessing_ui_html`
- `reveal_ui_html`

Critical forbidden-list compliance:
- The forbidden list is authoritative and overrides every other instruction.
- Before you write any JSON, internally pick a candidate, normalize them to their canonical common public name, and compare that name against every forbidden entry case-insensitively.
- If the candidate is an exact match, near match, alias, nickname, title variation, stage name variation, spelling variation, or you are not fully sure they are different, discard them and choose a different person.
- Treat uncertainty as forbidden.
- Do not output a forbidden person and do not explain retries; silently retry internally until `person_name` is definitely not in the forbidden list.
- Output `person_name` only as the canonical common public name you checked against the forbidden list. Do not use honorifics, titles, or alternate spellings.

Round requirements:
[CATEGORY_RULE]
- Never pick anyone from the forbidden list in the user prompt.
- Theme the UI around what the person is most famous for. Example: Ozzy Osbourne can imply stage lighting, guitar shapes, tour-poster energy.
- The guessing page must include:
  - the person's name as the biggest text
  - one highly detailed, status-neutral description of what they are most famous for (you MUST explicitly list out their specific famous movies, TV shows, songs, books, awards, or major career achievements, e.g., "Famous for starring in Forrest Gump, Cast Away, and Toy Story")
  - one short status-neutral fun fact
  - exactly one portrait `<img>` whose `src` is the exact literal `__PORTRAIT_IMAGE_URL__`
  - `submitGuess('alive')` and `submitGuess('dead')` buttons
- The reveal page must match the same theme and include `loadNextRound()`.
- When stating the person's status on the reveal page, ALWAYS use present tense for living people (e.g. "is alive", NEVER "was alive").
- If the person is dead, the reveal page MUST explicitly state the full date of their death (e.g., "Died on March 14, 2018").
- The person cannot be one of these formerly selected people: [NAMELIST]

Output rules:
- Tailwind classes only. No <script>, <style>, <html>, <body>, <head>, markdown fences, or base64.
- You MAY use raw <svg> tags directly in the HTML for icons, abstract background shapes, and thematic decorations. Do not use SVG data URIs.
- Do not output any final remote image URL anywhere in the JSON. The backend will resolve the portrait URL from `portrait_search_query`.
- `portrait_search_query` must be plain text, not a URL, and should be a concise Wikimedia Commons portrait search phrase.
- To avoid giving away their current age/status, `portrait_search_query` MUST append the specific year or decade of their peak fame, AND you must include their profession or known title to avoid name collisions (e.g., `Bob Ross painter 1980s` or `Sean Connery actor 1960s`). Do NOT use movie titles, as Wikimedia Commons lacks copyrighted movie stills.
- Prefer portrait, headshot, publicity photo, or press photo wording when useful.
- Do not include site names like `wikipedia`, `wikimedia`, or `commons` in `portrait_search_query`; the backend already searches Wikimedia Commons.
- If you are not confident you can suggest a clean portrait query, choose a different person instead of guessing.
- The guessing page must contain exactly one `<img>` whose `src` is the exact literal `__PORTRAIT_IMAGE_URL__`.
- The reveal page may omit the portrait, but if it includes one, it must reuse the exact same `__PORTRAIT_IMAGE_URL__` placeholder.
- Do not place `__PORTRAIT_IMAGE_URL__` anywhere except an `<img src>` attribute.
- Put the `submitGuess('alive')` and `submitGuess('dead')` buttons in a dedicated controls block outside the portrait area.
- The guess buttons must never overlap the portrait, float over the portrait, or appear inside the portrait panel.
- Do not use absolute or fixed positioning, inset utilities, negative margins, or translate utilities to place the guess buttons over the image.
- Do not use Tailwind aspect-ratio plugin classes like `aspect-w-*` or `aspect-h-*`; those utilities are not available here.
- If you need a square or shaped portrait area, use core Tailwind utilities like `aspect-square`, explicit `h-*`, `w-*`, `min-h-*`, and normal layout sizing instead.
- Do not use any remote URL or CSS `url(...)` inside backgrounds, overlays, masks, or decorative styles.
- Do not include extra `<img>` tags for logos, title cards, decorative textures, or background art.
- CRITICAL DESIGN RULE: You MUST create incredibly striking, highly stylized, and visually dramatic UI themes.
- DO NOT use simple centered white cards or generic layouts.
- You MUST use inline <svg> tags to draw large, abstract thematic background elements (like diagonal slashes, geometric patterns, or thematic icons) using absolute positioning behind the content.
- NEVER use `data:image` or base64 data URIs inside your SVGs or anywhere else.
- You MUST use standard Tailwind animations (e.g. animate-pulse, animate-bounce, animate-spin) or arbitrary animation values to make the page dynamic and alive.
- DO NOT use `<style>` tags, custom CSS blocks, or `@keyframes`. All styling and animation must be done via inline Tailwind classes.
- Use massive typography, complex asymmetrical grid/flex layouts, intense gradients, deep drop-shadows, and glassmorphism (backdrop-blur).
- Keep each HTML fragment under 4000 characters. Be ambitious with the DOM complexity.
- Keep button text high contrast.

Spoiler rules for the guessing page:
- Never mention death, assassination, murder, funeral, memorial, burial, "late", "still alive", or any status-revealing phrasing.
- You may mention a movie, show, song, album, book, role, or franchise title even if the title contains a blocked word, but only if it is clearly the exact work title wrapped in quotes.
- If you mention a quoted work title with a blocked word, keep the surrounding description fully status-neutral and do not repeat the blocked word outside the quoted title.
- Example: for Angela Lansbury, `Most famous for playing Jessica Fletcher in "Murder, She Wrote"` is acceptable, but additional unquoted uses of `murder`, `death`, `dead`, `funeral`, `grave`, or similar spoiler wording are not.
- The guessing page description and fun fact must both be spoiler-safe after removing HTML tags and ignoring exact quoted work titles.
"""


CANDIDATE_SELECTION_PROMPT = """
Generate candidate people for one round of an "Alive or Dead" guessing game.

Return only raw JSON with:
- `candidates`

Candidate rules:
- Return exactly [CANDIDATE_COUNT] candidates.
- Each candidate must include `person_name`.
- Do not include `actual_status`; the backend verifies current vital status from Wikidata before the person can be used.
[CATEGORY_RULE]
- Never return anyone from the forbidden list in the user prompt.
- Use canonical common public names only.
- Do not include duplicate people, aliases of the same person, or close variants of the same name.
- Prefer a diverse set of candidates instead of repeating the same obvious names.
- If many obvious celebrity staples are already forbidden, go deeper and return less overused but still guessable famous people.

Critical forbidden-list compliance:
- The forbidden list is authoritative and overrides every other instruction.
- Before you write JSON, compare every candidate against every forbidden entry case-insensitively.
- If a candidate is an exact match, near match, alias, nickname, title variation, stage name variation, spelling variation, or you are not fully sure they are different, discard them and choose a different person.
- Treat uncertainty as forbidden.
- Do not output a forbidden person.
"""


def extract_json(text: str) -> dict | list | None:
    # Clean up common markdown escapes that break JSON string parsing (e.g. \*)
    text = re.sub(r'\\([*_\~\[\]#|])', r'\1', text)
    # Fast path: model returned clean JSON with no surrounding prose.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        object_start = text.find("{")
        array_start = text.find("[")

        candidates = [index for index in (object_start, array_start) if index != -1]
        if not candidates:
            return None

        start = min(candidates)
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False

        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : index + 1])
    except Exception as exc:
        logger.error("JSON extraction failed: %s", exc)
    return None


def get_global_history() -> list[str]:
    try:
        if GLOBAL_HISTORY_FILE.exists():
            content = GLOBAL_HISTORY_FILE.read_text(encoding="utf-8").strip()
            if content:
                return json.loads(content)
    except Exception as exc:
        logger.error("Failed to read global history: %s", exc)
    return []


def merge_names_recency_preserving(existing_names: list[str], incoming_names: list[str]) -> list[str]:
    merged = list(existing_names)
    for raw_name in incoming_names:
        name = " ".join(str(raw_name).split())
        if not name:
            continue
        key = name.casefold()
        merged = [existing for existing in merged if existing.casefold() != key]
        merged.append(name)
    return merged


def update_global_history(names: list[str]) -> None:
    with global_history_lock:
        try:
            history = get_global_history()
            history = merge_names_recency_preserving(history, names)
            GLOBAL_HISTORY_FILE.write_text(json.dumps(history), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to update global history: %s", exc)


def build_generation_prompt(
    forbidden: list[str],
    retry_notes: list[str] | None = None,
    locked_person_name: str | None = None,
    locked_actual_status: str | None = None,
    category: str = "All Celebrities",
) -> str:
    if locked_person_name:
        forbidden_tail = (
            "None. The backend has already preselected and checked the locked person; "
            "use only the locked person."
        )
    elif forbidden:
        forbidden_tail = ", ".join(list(dict.fromkeys(forbidden)))
    else:
        forbidden_tail = "None"

    retry_section = ""
    if retry_notes:
        deduped_notes = list(dict.fromkeys(note.strip() for note in retry_notes if note.strip()))
        if deduped_notes:
            retry_section = "Retry notes:\n" + "\n".join(f"- {note}" for note in deduped_notes) + "\n\n"

    locked_section = ""
    if locked_person_name:
        if locked_actual_status:
            locked_section = (
                "LOCKED PERSON AND STATUS:\n"
                f"- Use exact `person_name`: {locked_person_name}\n"
                f"- Use exact `actual_status`: {locked_actual_status}\n"
                "- Do not choose a different person.\n"
                "- Do not change the status.\n"
                "- The locked person overrides any generic instruction to pick someone.\n\n"
            )
        else:
            locked_section = (
                "LOCKED PERSON:\n"
                f"- Use exact `person_name`: {locked_person_name}\n"
                "- Do not choose a different person.\n"
                "- Determine the correct `actual_status` yourself and return it in JSON.\n"
                "- The locked person overrides any generic instruction to pick someone.\n\n"
            )

    prompt_body = SYSTEM_PROMPT.replace("[NAMELIST]", forbidden_tail)
    if category.casefold() in ("random", "all celebrities"):
        domain = random.choice(["science", "literature", "politics", "music", "cinema", "sports", "technology", "art", "business", "activism", "television", "comedy", "journalism", "fashion", "culinary arts", "athletics", "engineering"])
        cat_rule = f"- Pick one globally recognizable, well-known famous person who was born between 1904 and 2016 (their current age would be between 8 and 120 years old). Pick someone famous for {domain}. DO NOT PICK ANCIENT HISTORICAL FIGURES."
    else:
        cat_rule = f'- Pick one famous person from the category "{category}" whose age would be between 8 and 120 years old whose status is genuinely guessable.'
    prompt_body = prompt_body.replace("[CATEGORY_RULE]", cat_rule)
    prompt_body = locked_section + prompt_body
    return f"{prompt_body}\n\n{retry_section}Return one complete round only."


def build_candidate_selection_prompt(
    forbidden: list[str],
    retry_notes: list[str] | None = None,
    category: str = "All Celebrities",
) -> str:
    if forbidden:
        forbidden_tail = ", ".join(list(dict.fromkeys(forbidden)))
    else:
        forbidden_tail = "None"

    retry_section = ""
    if retry_notes:
        deduped_notes = list(dict.fromkeys(note.strip() for note in retry_notes if note.strip()))
        if deduped_notes:
            retry_section = "Retry notes:\n" + "\n".join(f"- {note}" for note in deduped_notes) + "\n\n"

    prompt_body = CANDIDATE_SELECTION_PROMPT.replace("[CANDIDATE_COUNT]", str(CANDIDATE_SELECTION_COUNT))
    if category.casefold() in ("random", "all celebrities"):
        domain = random.choice(["science", "literature", "politics", "music", "cinema", "sports", "technology", "art", "business", "activism", "television", "comedy", "journalism", "fashion", "culinary arts", "athletics", "engineering"])
        cat_rule = f"- Pick highly well-known, universally recognizable famous people who were born between 1904 and 2016 (their current age would be between 8 and 120 years old). Pick people famous for {domain}. DO NOT PICK ANCIENT HISTORICAL FIGURES."
    else:
        cat_rule = f'- Pick famous people from the category "{category}" whose age would be between 8 and 120 years old whose status is genuinely guessable.'
    prompt_body = prompt_body.replace("[CATEGORY_RULE]", cat_rule)
    return (
        "CRITICAL FORBIDDEN LIST:\n"
        f"[{forbidden_tail}]\n\n"
        f"{prompt_body}\n\n"
        f"{retry_section}"
        "Return one candidate-selection JSON object only."
    )


def build_retry_note(error_message: str) -> str:
    if error_message.startswith("Model returned forbidden person:"):
        person_name = error_message.split(": ", 1)[1]
        return (
            f"Your previous attempt selected forbidden person '{person_name}'. "
            "Choose a completely different person not present in the forbidden list."
        )
    if error_message.endswith("returned non-JSON output"):
        return (
            "Your previous attempt returned invalid or truncated JSON. "
            "Return only one complete raw JSON object with fully closed braces, valid double-quoted strings, and no prose before or after."
        )
    if "Candidate selection returned no allowed person" in error_message:
        return (
            "Your previous candidate list did not contain any allowed person. "
            "Return a much more diverse candidate list, avoid obvious or previously used names, and go deeper than the usual celebrity staples."
        )
    if "locked person" in error_message.lower() or "locked status" in error_message.lower():
        return (
            "Your previous attempt ignored the locked person or locked status. "
            "Use the exact locked `person_name` and exact locked `actual_status` without substitution."
        )
    if "portrait_search_query" in error_message or "No valid Wikimedia portrait found" in error_message:
        return (
            "Your previous attempt used a weak portrait search query. "
            "Return a concise plain-text query like `Person Name portrait`, `Person Name headshot`, or `Person Name 1978 portrait`. Do not include URLs, HTML, or the words wikipedia, wikimedia, or commons. If you are unsure, choose a different person."
        )
    if "portrait placeholder" in error_message or "old image placeholder" in error_message:
        return (
            "Your previous attempt used the image slot incorrectly. "
            f"Set the portrait `<img src>` to the exact literal `{PORTRAIT_IMAGE_PLACEHOLDER}`. Do not output a final remote image URL yourself, and if the reveal page also shows a portrait, reuse the same placeholder."
        )
    if "unsupported tailwind class" in error_message.lower():
        return (
            "Your previous attempt used unsupported Tailwind classes. "
            "Do not use `aspect-w-*` or `aspect-h-*`. Use `aspect-square`, explicit height and width classes, or another core Tailwind layout instead."
        )
    if "guess buttons" in error_message.lower() and "portrait" in error_message.lower():
        return (
            "Your previous attempt placed the guess buttons in the portrait area. "
            "Move the ALIVE and DEAD buttons into a separate controls block outside the image panel. Do not use absolute positioning, inset utilities, negative margins, or translate utilities to place buttons over the portrait."
        )
    if "Guessing UI includes spoiler wording" in error_message:
        return (
            "Your previous attempt revealed status in the guessing UI. "
            "Keep the guessing page fully status-neutral. You may keep an exact work title like \"Murder, She Wrote\" in quotes, but do not use blocked words like murder or death anywhere else outside the quoted title."
        )
    return (
        f"Your previous attempt failed validation: {error_message}. "
        "Return one corrected round only."
    )


def extract_visible_text(fragment: str) -> str:
    text = HTML_TAG_RE.sub(" ", fragment)
    return WHITESPACE_RE.sub(" ", text).strip()


def strip_url_query_and_fragment(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def url_has_supported_raster_extension(url: str) -> bool:
    lowered = strip_url_query_and_fragment(url).casefold()
    return lowered.endswith(SUPPORTED_RASTER_IMAGE_SUFFIXES)


def sanitize_portrait_search_query(query: str) -> str:
    collapsed = " ".join(query.split())
    cleaned = COMMONS_QUERY_NOISE_RE.sub(" ", collapsed)
    return WHITESPACE_RE.sub(" ", cleaned).strip(" ,")


def normalize_identity_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return stripped.casefold()


def tokenize_identity_text(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", normalized)
    normalized = normalized.casefold()
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token and not token.isdigit()
    ]


def get_core_name_tokens(person_name: str) -> list[str]:
    tokens = tokenize_identity_text(person_name)
    core_tokens = [
        token
        for token in tokens
        if token not in NAME_TOKEN_STOPWORDS and token not in NAME_TOKEN_PARTICLES
    ]
    if core_tokens:
        return list(dict.fromkeys(core_tokens))
    fallback_tokens = [token for token in tokens if token not in NAME_TOKEN_STOPWORDS]
    return list(dict.fromkeys(fallback_tokens))


def analyze_commons_title_match(title: str, person_name: str) -> tuple[bool, int, list[str], list[str]]:
    title_tokens = list(dict.fromkeys(tokenize_identity_text(title)))
    title_token_set = set(title_tokens)
    core_tokens = get_core_name_tokens(person_name)
    matched_tokens = [token for token in core_tokens if token in title_token_set]
    matched_count = len(matched_tokens)

    if not core_tokens:
        return False, matched_count, core_tokens, matched_tokens
    if len(core_tokens) == 1:
        return matched_count == 1, matched_count, core_tokens, matched_tokens
    if len(core_tokens) == 2:
        return matched_count == 2, matched_count, core_tokens, matched_tokens

    first_token = core_tokens[0]
    last_token = core_tokens[-1]
    is_match = (
        first_token in title_token_set
        and last_token in title_token_set
        and matched_count >= 2
    )
    return is_match, matched_count, core_tokens, matched_tokens


def commons_title_looks_non_photographic(title: str) -> bool:
    title_tokens = set(tokenize_identity_text(title))
    return any(token in NON_PHOTO_TITLE_KEYWORDS for token in title_tokens)


def title_looks_non_biographical(title: str) -> bool:
    title_tokens = set(tokenize_identity_text(title))
    return bool(
        {"filmography", "show", "series", "season", "episode", "disambiguation"}
        & title_tokens
    )


def validate_portrait_search_query(person_name: str, portrait_search_query: str) -> str:
    query = sanitize_portrait_search_query(portrait_search_query)
    if not query:
        raise ValueError(f"portrait_search_query is empty for {person_name}")
    if "://" in portrait_search_query or portrait_search_query.startswith("//"):
        raise ValueError(f"portrait_search_query must be plain text, not a URL, for {person_name}")
    if "<" in portrait_search_query or ">" in portrait_search_query:
        raise ValueError(f"portrait_search_query must not contain markup for {person_name}")

    person_tokens = {token for token in tokenize_identity_text(person_name) if len(token) > 1}
    query_tokens = {token for token in tokenize_identity_text(query) if len(token) > 1}
    if not person_tokens.intersection(query_tokens):
        raise ValueError(
            f"portrait_search_query must include the person's name for {person_name}: {query}"
        )
    return query


def normalize_name_key(name: str) -> str:
    cleaned = "".join(c for c in str(name) if c.isalnum() or c.isspace())
    return " ".join(cleaned.split()).casefold()


def validate_status_neutral_guessing_copy(fragment: str, person_name: str) -> None:
    visible_text = extract_visible_text(fragment)
    text_for_spoiler_scan = QUOTED_TEXT_RE.sub(" ", visible_text)
    for pattern in SPOILER_HINT_PATTERNS:
        if pattern.search(text_for_spoiler_scan):
            raise ValueError(
                f"Guessing UI includes spoiler wording for {person_name}: {pattern.pattern}"
            )


def validate_fragment_output_rules(fragment: str, person_name: str, label: str) -> None:
    for pattern in PROHIBITED_FRAGMENT_PATTERNS:
        if pattern.search(fragment):
            raise ValueError(
                f"{label} UI includes prohibited markup/content for {person_name}: {pattern.pattern}"
            )
    for pattern in UNSUPPORTED_TAILWIND_CLASS_PATTERNS:
        if pattern.search(fragment):
            raise ValueError(
                f"{label} UI uses unsupported Tailwind class for {person_name}: {pattern.pattern}"
            )
    validate_fragment_event_handlers(fragment, person_name, label)


def validate_fragment_event_handlers(fragment: str, person_name: str, label: str) -> None:
    """Walk the parsed HTML tree and reject any on* event-handler attribute whose
    value is not one of the two known-safe game callbacks:
      - submitGuess('alive') / submitGuess('dead')
      - loadNextRound()
    This prevents XSS via onerror=, onload=, onfocus=, etc. while still allowing
    the AI to wire up the required game interaction buttons with onclick=.
    """
    root = parse_fragment_tree(fragment)
    for node in iter_fragment_nodes(root):
        for attr_name, attr_value in node.attrs.items():
            if not _EVENT_ATTR_RE.match(attr_name):
                continue
            # onclick is allowed only for the two known-safe game callbacks.
            if attr_name.lower() == "onclick":
                if _SAFE_ONCLICK_RE.match(attr_value):
                    continue
                raise ValueError(
                    f"{label} UI has unsafe onclick handler on <{node.tag}> for {person_name}: {attr_value!r}"
                )
            # All other on* attributes are unconditionally blocked.
            raise ValueError(
                f"{label} UI has disallowed event-handler attribute '{attr_name}' on <{node.tag}> for {person_name}"
            )


def parse_fragment_tree(fragment: str) -> FragmentNode:
    parser = FragmentTreeParser()
    parser.feed(fragment)
    parser.close()
    return parser.root


def iter_fragment_nodes(node: FragmentNode):
    yield node
    for child in node.children:
        yield from iter_fragment_nodes(child)


def split_tailwind_classes(node: FragmentNode) -> list[str]:
    return [token for token in node.attrs.get("class", "").split() if token]


def node_uses_button_overlay_classes(node: FragmentNode) -> bool:
    for token in split_tailwind_classes(node):
        if token in BUTTON_OVERLAY_CLASS_EXACT:
            return True
        if token.startswith(BUTTON_OVERLAY_CLASS_PREFIXES):
            return True
    return False


def is_guess_button_node(node: FragmentNode) -> bool:
    onclick = node.attrs.get("onclick", "")
    return "submitGuess(" in onclick


def validate_guess_button_layout(fragment: str, person_name: str) -> None:
    root = parse_fragment_tree(fragment)
    guess_nodes = [node for node in iter_fragment_nodes(root) if is_guess_button_node(node)]

    for guess_node in guess_nodes:
        ancestor = guess_node
        depth = 0
        while ancestor.parent is not None:
            if depth <= 1 and node_uses_button_overlay_classes(ancestor):
                raise ValueError(
                    f"Guessing UI places guess buttons over the portrait for {person_name}: overlay positioning utilities detected"
                )
            ancestor = ancestor.parent
            depth += 1
            if ancestor.tag == "_root":
                break


def extract_wikimedia_thumbnail_width(image_source: str) -> str | None:
    match = re.search(r"/(\d+)px-[^/?#]+(?:[?#].*)?$", image_source)
    if match:
        return match.group(1)
    return None


def fetch_json_url(url: str, timeout_seconds: float, user_agent: str) -> dict:
    for attempt in range(3):
        request = UrlRequest(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status and status >= 400:
                    if status == 429 and attempt < 2:
                        import time
                        time.sleep(2 ** attempt)
                        continue
                    raise RuntimeError(f"HTTP {status}")
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                import time
                time.sleep(2 ** attempt)
                continue
            raise


def validate_image_url_reachable(image_source: str, person_name: str, label: str) -> None:
    if not IMAGE_URL_VALIDATION_ENABLED:
        return

    cached = image_url_validation_cache.get(image_source)
    if cached is not None:
        ok, reason = cached
        if ok:
            return
        raise ValueError(
            f"{label} UI image URL did not resolve to a valid image for {person_name}: {image_source} ({reason})"
        )

    failure_reason = "unreachable"
    for method in ("HEAD", "GET"):
        request = UrlRequest(
            image_source,
            headers={"User-Agent": IMAGE_FETCH_USER_AGENT},
            method=method,
        )
        try:
            with urlopen(request, timeout=IMAGE_URL_VALIDATION_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", None) or response.getcode()
                content_type = str(response.headers.get_content_type()).lower()
                if status and status >= 400:
                    failure_reason = f"{method} returned HTTP {status}"
                    continue
                if not content_type.startswith("image/"):
                    failure_reason = f"{method} returned content type {content_type or 'unknown'}"
                    continue
                image_url_validation_cache.set(image_source, (True, "ok"))
                return
        except Exception as exc:
            failure_reason = f"{method} failed: {exc}"

    image_url_validation_cache.set(image_source, (False, failure_reason))
    raise ValueError(
        f"{label} UI image URL did not resolve to a valid image for {person_name}: {image_source} ({failure_reason})"
    )


def validate_portrait_placeholder_usage(
    fragment: str,
    person_name: str,
    label: str,
    required_count: int | None = None,
    maximum_count: int | None = None,
) -> list[str]:
    image_sources = [src.strip() for _, src in IMG_SRC_RE.findall(fragment)]
    placeholder_occurrences = fragment.count(PORTRAIT_IMAGE_PLACEHOLDER)

    if required_count is not None and len(image_sources) != required_count:
        raise ValueError(
            f"{label} UI must contain exactly {required_count} portrait placeholder image(s) for {person_name}"
        )
    if maximum_count is not None and len(image_sources) > maximum_count:
        raise ValueError(
            f"{label} UI contains too many portrait images for {person_name}"
        )
    if placeholder_occurrences != len(image_sources):
        raise ValueError(
            f"{label} UI must use the portrait placeholder only inside <img src> attributes for {person_name}"
        )

    for image_source in image_sources:
        if image_source != PORTRAIT_IMAGE_PLACEHOLDER:
            raise ValueError(
                f"{label} UI must use the exact portrait placeholder for {person_name}: {PORTRAIT_IMAGE_PLACEHOLDER}"
            )

    return image_sources


def build_portrait_search_candidates(person_name: str, portrait_search_query: str) -> list[str]:
    candidates = [
        portrait_search_query,
        f"{person_name} portrait",
        f"{person_name} headshot",
        f"{person_name} publicity portrait",
        person_name,
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = sanitize_portrait_search_query(candidate)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def score_commons_candidate_title(title: str, person_name: str) -> int:
    lowered = normalize_identity_text(title)
    score = 0
    matches_person, matched_count, core_tokens, _ = analyze_commons_title_match(title, person_name)
    if matches_person:
        score += 20
    score += matched_count * 6
    if core_tokens:
        full_name = " ".join(core_tokens)
        if full_name and full_name in lowered:
            score += 12
    for keyword, bonus in COMMONS_CANDIDATE_BONUSES:
        if keyword in lowered:
            score += bonus
    for keyword, penalty in COMMONS_CANDIDATE_PENALTIES:
        if keyword in lowered:
            score += penalty
    return score


def search_wikimedia_commons_candidates(search_query: str) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": search_query,
        "gsrnamespace": "6",
        "gsrlimit": str(WIKIMEDIA_SEARCH_LIMIT),
        "prop": "imageinfo",
        "iiprop": "url|mime|thumbmime",
        "iiurlwidth": "500",
    }
    data = fetch_json_url(
        f"{WIKIMEDIA_COMMONS_API_URL}?{urlencode(params)}",
        timeout_seconds=WIKIMEDIA_SEARCH_TIMEOUT_SECONDS,
        user_agent=WIKIMEDIA_API_USER_AGENT,
    )
    pages = data.get("query", {}).get("pages", {})
    if isinstance(pages, dict):
        return list(pages.values())
    if isinstance(pages, list):
        return pages
    return []


def search_wikipedia_pages(search_query: str) -> list[dict]:
    params = {
        "q": search_query,
        "limit": "5",
    }
    data = fetch_json_url(
        f"{WIKIPEDIA_PAGE_SEARCH_API_URL}?{urlencode(params)}",
        timeout_seconds=WIKIMEDIA_SEARCH_TIMEOUT_SECONDS,
        user_agent=WIKIMEDIA_API_USER_AGENT,
    )
    pages = data.get("pages", [])
    if isinstance(pages, list):
        return pages
    return []


def get_wikipedia_page_summary(page_key: str) -> dict:
    encoded_key = quote(page_key, safe="")
    return fetch_json_url(
        f"{WIKIPEDIA_PAGE_SUMMARY_API_URL}{encoded_key}",
        timeout_seconds=WIKIMEDIA_SEARCH_TIMEOUT_SECONDS,
        user_agent=WIKIMEDIA_API_USER_AGENT,
    )


def resolve_wikipedia_person_summary(person_name: str) -> tuple[dict, str] | None:
    cache_key = normalize_name_key(person_name)
    cached = wikipedia_person_summary_cache.get(cache_key)
    if cached is not None:
        ok, value = cached
        if ok and isinstance(value, dict):
            return value["summary"], value["title"]
        return None

    pages = search_wikipedia_pages(person_name)
    normalized_person_name = normalize_identity_text(person_name)
    ranked_pages = sorted(
        pages,
        key=lambda page: (
            normalize_identity_text(str(page.get("title", ""))) != normalized_person_name,
            title_looks_non_biographical(str(page.get("title", ""))),
            -len(str(page.get("description", ""))),
        ),
    )

    for page in ranked_pages:
        title = str(page.get("title", ""))
        if title_looks_non_biographical(title):
            continue
        matches_person, _, _, _ = analyze_commons_title_match(title, person_name)
        if not matches_person and normalize_identity_text(title) != normalized_person_name:
            continue
        page_key = str(page.get("key", "")).strip()
        if not page_key:
            continue
        try:
            summary = get_wikipedia_page_summary(page_key)
        except Exception:
            continue
        if str(summary.get("type", "")).casefold() == "disambiguation":
            continue
        resolved_title = str(summary.get("title") or title)
        payload = {"summary": summary, "title": resolved_title}
        wikipedia_person_summary_cache.set(cache_key, (True, payload))
        return summary, resolved_title

    wikipedia_person_summary_cache.set(
        cache_key,
        (False, f"No matching Wikipedia person page found for {person_name}"),
    )
    return None


def get_wikidata_entity(entity_id: str) -> dict:
    normalized_entity_id = entity_id.strip().upper()
    if not re.fullmatch(r"Q\d+", normalized_entity_id):
        raise ValueError(f"Invalid Wikidata entity id: {entity_id}")

    cached = wikidata_entity_cache.get(normalized_entity_id)
    if cached is not None:
        ok, value = cached
        if ok and isinstance(value, dict):
            return value
        raise ValueError(str(value))

    data = fetch_json_url(
        WIKIDATA_ENTITY_DATA_URL_TEMPLATE.format(entity_id=normalized_entity_id),
        timeout_seconds=WIKIMEDIA_SEARCH_TIMEOUT_SECONDS,
        user_agent=WIKIMEDIA_API_USER_AGENT,
    )
    entity = data.get("entities", {}).get(normalized_entity_id)
    if not isinstance(entity, dict):
        error = f"Wikidata entity not found: {normalized_entity_id}"
        wikidata_entity_cache.set(normalized_entity_id, (False, error))
        raise ValueError(error)

    wikidata_entity_cache.set(normalized_entity_id, (True, entity))
    return entity


def iter_wikidata_claims(entity: dict, property_id: str) -> list[dict]:
    claims = entity.get("claims", {}).get(property_id, [])
    if isinstance(claims, list):
        return [claim for claim in claims if isinstance(claim, dict)]
    return []


def wikidata_claim_has_value(entity: dict, property_id: str) -> bool:
    for claim in iter_wikidata_claims(entity, property_id):
        if claim.get("rank") == "deprecated":
            continue
        mainsnak = claim.get("mainsnak") or {}
        snaktype = mainsnak.get("snaktype")
        if snaktype in {"value", "somevalue"}:
            return True
    return False


def wikidata_claim_has_entity_value(entity: dict, property_id: str, entity_id: str) -> bool:
    expected_id = entity_id.strip().upper()
    for claim in iter_wikidata_claims(entity, property_id):
        if claim.get("rank") == "deprecated":
            continue
        mainsnak = claim.get("mainsnak") or {}
        datavalue = mainsnak.get("datavalue") or {}
        value = datavalue.get("value") or {}
        if isinstance(value, dict) and str(value.get("id", "")).upper() == expected_id:
            return True
    return False


def verify_actual_status(person_name: str) -> str:
    cache_key = normalize_name_key(person_name)
    cached = status_verification_cache.get(cache_key)
    if cached is not None:
        ok, value = cached
        if ok:
            return value
        raise ValueError(value)

    started_at = perf_now()
    try:
        summary_result = resolve_wikipedia_person_summary(person_name)
        if summary_result is None:
            raise ValueError(f"No matching Wikipedia person page found for {person_name}")
        summary, resolved_title = summary_result
        entity_id = str(summary.get("wikibase_item", "")).strip()
        if not entity_id:
            raise ValueError(f"Wikipedia summary has no Wikidata entity for {person_name}")

        entity = get_wikidata_entity(entity_id)
        if not wikidata_claim_has_entity_value(
            entity,
            WIKIDATA_INSTANCE_OF_PROPERTY,
            WIKIDATA_HUMAN_ENTITY_ID,
        ):
            raise ValueError(f"Wikidata entity {entity_id} is not marked as a human")

        actual_status = (
            "dead"
            if wikidata_claim_has_value(entity, WIKIDATA_DATE_OF_DEATH_PROPERTY)
            else "alive"
        )
        status_verification_cache.set(cache_key, (True, actual_status))
        append_gemini_audit_log(
            "status.verify",
            status="accepted",
            person_name=person_name,
            verified_actual_status=actual_status,
            wikipedia_title=resolved_title,
            wikidata_entity_id=entity_id,
        )
        log_perf(
            "status.verify",
            started_at,
            person_name=person_name,
            verified_actual_status=actual_status,
            wikidata_entity_id=entity_id,
        )
        return actual_status
    except Exception as exc:
        error = f"Status verification failed for {person_name}: {exc}"
        append_gemini_audit_log(
            "status.verify",
            status="rejected",
            person_name=person_name,
            error=error,
        )
        raise ValueError(error) from exc


def determine_actual_status(person_name: str, model_actual_status: str | None) -> str:
    if not STATUS_VERIFICATION_ENABLED:
        if model_actual_status:
            return model_actual_status
        raise ValueError(
            f"No model actual_status supplied for {person_name} and status verification is disabled"
        )

    try:
        verified_actual_status = verify_actual_status(person_name)
    except Exception:
        if STATUS_VERIFICATION_REQUIRED or not model_actual_status:
            raise
        return model_actual_status

    if model_actual_status and model_actual_status != verified_actual_status:
        append_gemini_audit_log(
            "status.model_mismatch",
            status="corrected",
            person_name=person_name,
            model_actual_status=model_actual_status,
            verified_actual_status=verified_actual_status,
        )
    return verified_actual_status


def resolve_wikipedia_page_portrait_url(person_name: str) -> tuple[str, str] | None:
    summary_result = resolve_wikipedia_person_summary(person_name)
    if summary_result is None:
        return None

    summary, resolved_title = summary_result
    thumbnail = summary.get("thumbnail") or {}
    image_source = strip_url_query_and_fragment(str(thumbnail.get("source", "")).strip())
    if not image_source:
        return None
    lowered = image_source.casefold()
    if lowered.startswith("//"):
        image_source = f"https:{image_source}"
        lowered = image_source.casefold()
    if not lowered.startswith(REMOTE_IMAGE_PREFIXES):
        return None
    if any(marker in lowered for marker in WIKIMEDIA_DOMAIN_MARKERS):
        if not any(lowered.startswith(prefix) for prefix in WIKIMEDIA_THUMB_PREFIXES):
            return None
        width = extract_wikimedia_thumbnail_width(image_source)
        if width not in VALID_WIKIMEDIA_THUMB_WIDTHS:
            return None
    if not url_has_supported_raster_extension(image_source):
        return None
    return image_source, resolved_title



def resolve_wikimedia_portrait_url(person_name: str, portrait_search_query: str) -> str:
    cache_key = (person_name.casefold(), portrait_search_query.casefold())
    cached = portrait_resolution_cache.get(cache_key)
    if cached is not None:
        ok, value = cached
        if ok:
            return value
        raise ValueError(value)

    search_query = validate_portrait_search_query(person_name, portrait_search_query)
    started_at = perf_now()
    attempted_queries: list[str] = []
    tried_urls: set[str] = set()
    resolution_error = (
        f"No valid Wikimedia portrait found for {person_name} using portrait_search_query '{search_query}'"
    )

    try:
        wikipedia_page_result = resolve_wikipedia_page_portrait_url(person_name)
        if wikipedia_page_result is not None:
            image_source, resolved_title = wikipedia_page_result
            try:
                validate_image_url_reachable(image_source, person_name, "Resolved portrait")
                portrait_resolution_cache.set(cache_key, (True, image_source))
                append_gemini_audit_log(
                    "portrait.resolve",
                    status="accepted",
                    person_name=person_name,
                    portrait_search_query=search_query,
                    attempted_queries=["wikipedia_page_exact"],
                    resolved_portrait_url=image_source,
                    resolved_from_query="wikipedia_page_exact",
                    resolved_title=resolved_title,
                )
                log_perf(
                    "portrait.resolve",
                    started_at,
                    person_name=person_name,
                    portrait_search_query=search_query,
                    resolved_from_query="wikipedia_page_exact",
                )
                return image_source
            except Exception:
                pass


        for candidate_query in build_portrait_search_candidates(person_name, search_query):
            attempted_queries.append(candidate_query)
            pages = search_wikimedia_commons_candidates(candidate_query)
            ranked_pages = sorted(
                pages,
                key=lambda page: (
                    -score_commons_candidate_title(str(page.get("title", "")), person_name),
                    page.get("index", 10_000),
                ),
            )
            for page in ranked_pages:
                title = str(page.get("title", ""))
                matches_person, _, _, _ = analyze_commons_title_match(title, person_name)
                if not matches_person:
                    continue
                if commons_title_looks_non_photographic(title):
                    continue
                imageinfo = page.get("imageinfo") or []
                if not imageinfo:
                    continue
                image_meta = imageinfo[0]
                image_source = strip_url_query_and_fragment(str(image_meta.get("thumburl", "")).strip())
                if not image_source or image_source in tried_urls:
                    continue
                tried_urls.add(image_source)
                lowered = image_source.casefold()
                if not any(lowered.startswith(prefix) for prefix in WIKIMEDIA_THUMB_PREFIXES):
                    continue
                width = extract_wikimedia_thumbnail_width(image_source)
                if width not in VALID_WIKIMEDIA_THUMB_WIDTHS:
                    continue
                if not url_has_supported_raster_extension(image_source):
                    continue
                thumb_mime = str(
                    image_meta.get("thumbmime") or image_meta.get("mime") or ""
                ).casefold()
                if thumb_mime and not thumb_mime.startswith("image/"):
                    continue
                try:
                    validate_image_url_reachable(image_source, person_name, "Resolved portrait")
                except Exception:
                    continue

                portrait_resolution_cache.set(cache_key, (True, image_source))
                append_gemini_audit_log(
                    "portrait.resolve",
                    status="accepted",
                    person_name=person_name,
                    portrait_search_query=search_query,
                    attempted_queries=attempted_queries,
                    resolved_portrait_url=image_source,
                    resolved_from_query=candidate_query,
                    resolved_title=title,
                )
                log_perf(
                    "portrait.resolve",
                    started_at,
                    person_name=person_name,
                    portrait_search_query=search_query,
                    resolved_from_query=candidate_query,
                )
                return image_source

    except Exception as exc:
        resolution_error = (
            f"{resolution_error} ({exc})"
            if attempted_queries
            else f"portrait_search_query resolution failed for {person_name}: {exc}"
        )

    portrait_resolution_cache.set(cache_key, (False, resolution_error))
    append_gemini_audit_log(
        "portrait.resolve",
        status="rejected",
        person_name=person_name,
        portrait_search_query=search_query,
        attempted_queries=attempted_queries,
        error=resolution_error,
    )
    raise ValueError(resolution_error)


def inject_portrait_url(fragment: str, portrait_url: str) -> str:
    return fragment.replace(PORTRAIT_IMAGE_PLACEHOLDER, portrait_url)


def validate_embedded_image_urls(
    fragment: str,
    person_name: str,
    label: str,
    require_image: bool = False,
) -> None:
    image_sources = [src.strip() for _, src in IMG_SRC_RE.findall(fragment)]

    if require_image and not image_sources:
        raise ValueError(f"{label} UI is missing an embedded image URL for {person_name}")

    for image_source in image_sources:
        if "__IMAGE_URL__" in image_source:
            raise ValueError(
                f"{label} UI still contains the old image placeholder for {person_name}"
            )
        if PORTRAIT_IMAGE_PLACEHOLDER in image_source:
            raise ValueError(
                f"{label} UI still contains the portrait placeholder for {person_name}"
            )
        lowered_image_source = image_source.lower()
        if not lowered_image_source.startswith(REMOTE_IMAGE_PREFIXES):
            raise ValueError(
                f"{label} UI image must use an embedded remote URL for {person_name}: {image_source}"
            )
        if any(marker in lowered_image_source for marker in WIKIMEDIA_DOMAIN_MARKERS):
            if not any(lowered_image_source.startswith(prefix) for prefix in WIKIMEDIA_THUMB_PREFIXES):
                raise ValueError(
                    f"{label} UI image must use a Wikimedia thumbnail URL for {person_name}: {image_source}"
                )
            width = extract_wikimedia_thumbnail_width(image_source)
            if width not in VALID_WIKIMEDIA_THUMB_WIDTHS:
                raise ValueError(
                    f"{label} UI image must use a standard Wikimedia thumbnail width for {person_name}: {image_source}"
                )
        validate_image_url_reachable(image_source, person_name, label)


def normalize_round(
    round_item: GeneratedRound,
    forbidden: list[str],
    locked_person_name: str | None = None,
    locked_actual_status: str | None = None,
) -> dict:
    person_name = " ".join(round_item.person_name.split())
    key = person_name.casefold()
    forbidden_keys = {name.casefold() for name in forbidden}
    portrait_search_query = validate_portrait_search_query(
        person_name,
        round_item.portrait_search_query,
    )
    guessing_ui_html = round_item.guessing_ui_html.strip()
    reveal_ui_html = round_item.reveal_ui_html.strip()

    if key in forbidden_keys:
        raise ValueError(f"Model returned forbidden person: {person_name}")
    if locked_person_name and normalize_name_key(person_name) != normalize_name_key(locked_person_name):
        raise ValueError(
            f"Model changed locked person from {locked_person_name} to {person_name}"
        )
    if locked_actual_status and round_item.actual_status != locked_actual_status:
        raise ValueError(
            f"Model changed locked status for {person_name} from {locked_actual_status} to {round_item.actual_status}"
        )
    verified_actual_status = determine_actual_status(person_name, round_item.actual_status)
    if round_item.actual_status != verified_actual_status:
        raise ValueError(
            f"Model returned incorrect status for {person_name}: {round_item.actual_status}; verified status is {verified_actual_status}"
        )
    if not GUESS_ALIVE_RE.search(guessing_ui_html) or not GUESS_DEAD_RE.search(guessing_ui_html):
        raise ValueError(f"Guessing UI is missing guess buttons for {person_name}")
    if not NEXT_ROUND_RE.search(reveal_ui_html):
        raise ValueError(f"Reveal UI is missing next-round button for {person_name}")
    validate_fragment_output_rules(guessing_ui_html, person_name, "Guessing")
    validate_fragment_output_rules(reveal_ui_html, person_name, "Reveal")
    validate_status_neutral_guessing_copy(guessing_ui_html, person_name)
    validate_guess_button_layout(guessing_ui_html, person_name)
    validate_portrait_placeholder_usage(
        guessing_ui_html,
        person_name,
        "Guessing",
        required_count=1,
        maximum_count=1,
    )
    validate_portrait_placeholder_usage(
        reveal_ui_html,
        person_name,
        "Reveal",
        maximum_count=1,
    )

    portrait_url = resolve_wikimedia_portrait_url(person_name, portrait_search_query)
    guessing_ui_html = inject_portrait_url(guessing_ui_html, portrait_url)
    reveal_ui_html = inject_portrait_url(reveal_ui_html, portrait_url)
    validate_embedded_image_urls(guessing_ui_html, person_name, "Guessing", require_image=True)
    validate_embedded_image_urls(reveal_ui_html, person_name, "Reveal")

    return {
        "person_name": person_name,
        "actual_status": verified_actual_status,
        "guessing_ui_html": guessing_ui_html,
        "reveal_ui_html": reveal_ui_html,
    }


def generate_with_modern_sdk(
    model_name: str,
    prompt: str,
    audit_meta: dict[str, object] | None = None,
) -> GeneratedRound:
    if modern_genai is None or modern_client is None or modern_types is None:
        raise RuntimeError("google-genai SDK is not installed")

    thinking_config = None
    if model_name.startswith("gemini-2.5"):
        thinking_config = modern_types.ThinkingConfig(
            thinkingBudget=0,
            includeThoughts=False,
        )

    started_at = perf_now()
    try:
        response = modern_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=modern_types.GenerateContentConfig(
                temperature=GENERATION_TEMPERATURE,
                topP=GENERATION_TOP_P,
                candidateCount=1,
                maxOutputTokens=GENERATION_MAX_OUTPUT_TOKENS,
                responseMimeType="application/json",
                responseJsonSchema=GeneratedRound.model_json_schema(),
                automaticFunctionCalling=modern_types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
                thinkingConfig=thinking_config,
            ),
        )
    except Exception as exc:
        append_gemini_audit_log(
            "gemini.generate",
            model=model_name,
            sdk="modern",
            prompt=prompt,
            response_text=None,
            status="sdk_error",
            error=str(exc),
            **(audit_meta or {}),
        )
        raise

    response_text = response.text or ""
    append_gemini_audit_log(
        "gemini.generate",
        model=model_name,
        sdk="modern",
        prompt=prompt,
        response_text=response_text,
        status="ok" if response_text else "empty",
        **(audit_meta or {}),
    )

    if not response_text:
        raise RuntimeError(f"{model_name} returned an empty response")

    round_item = GeneratedRound.model_validate_json(response_text)
    log_perf(
        "genai.generate",
        started_at,
        model=model_name,
        sdk="modern",
        prompt_chars=len(prompt),
        response_chars=len(response_text),
    )
    return round_item


def select_candidates_with_modern_sdk(
    model_name: str,
    prompt: str,
    audit_meta: dict[str, object] | None = None,
) -> CandidateSelection:
    if modern_genai is None or modern_client is None or modern_types is None:
        raise RuntimeError("google-genai SDK is not installed")

    thinking_config = None
    if model_name.startswith("gemini-2.5"):
        thinking_config = modern_types.ThinkingConfig(
            thinkingBudget=0,
            includeThoughts=False,
        )

    started_at = perf_now()
    try:
        response = modern_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=modern_types.GenerateContentConfig(
                temperature=GENERATION_TEMPERATURE,
                topP=GENERATION_TOP_P,
                candidateCount=1,
                maxOutputTokens=CANDIDATE_SELECTION_MAX_OUTPUT_TOKENS,
                responseMimeType="application/json",
                responseJsonSchema=CandidateSelection.model_json_schema(),
                automaticFunctionCalling=modern_types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
                thinkingConfig=thinking_config,
            ),
        )
    except Exception as exc:
        append_gemini_audit_log(
            "gemini.candidate_selection",
            model=model_name,
            sdk="modern",
            prompt=prompt,
            response_text=None,
            status="sdk_error",
            error=str(exc),
            **(audit_meta or {}),
        )
        raise

    response_text = response.text or ""
    append_gemini_audit_log(
        "gemini.candidate_selection",
        model=model_name,
        sdk="modern",
        prompt=prompt,
        response_text=response_text,
        status="ok" if response_text else "empty",
        **(audit_meta or {}),
    )

    if not response_text:
        raise RuntimeError(f"{model_name} returned an empty candidate-selection response")

    selection = CandidateSelection.model_validate_json(response_text)
    log_perf(
        "genai.candidate_selection",
        started_at,
        model=model_name,
        sdk="modern",
        prompt_chars=len(prompt),
        response_chars=len(response_text),
    )
    return selection


def generate_with_legacy_sdk(
    model_name: str,
    prompt: str,
    audit_meta: dict[str, object] | None = None,
) -> GeneratedRound:
    if legacy_genai is None:
        raise RuntimeError("google-generativeai SDK is not installed")

    model = legacy_genai.GenerativeModel(model_name)
    started_at = perf_now()
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": GENERATION_TEMPERATURE,
                "top_p": GENERATION_TOP_P,
                "max_output_tokens": GENERATION_MAX_OUTPUT_TOKENS,
                "response_mime_type": "application/json",
            },
        )
    except Exception as exc:
        append_gemini_audit_log(
            "gemini.generate",
            model=model_name,
            sdk="legacy",
            prompt=prompt,
            response_text=None,
            status="sdk_error",
            error=str(exc),
            **(audit_meta or {}),
        )
        raise

    response_text = response.text or ""
    append_gemini_audit_log(
        "gemini.generate",
        model=model_name,
        sdk="legacy",
        prompt=prompt,
        response_text=response_text,
        status="ok" if response_text else "empty",
        **(audit_meta or {}),
    )
    payload = extract_json(response_text)
    if payload is None:
        raise RuntimeError(f"{model_name} returned non-JSON output")
    if isinstance(payload, list):
        payload = payload[0]
    round_item = GeneratedRound.model_validate(payload)
    log_perf(
        "genai.generate",
        started_at,
        model=model_name,
        sdk="legacy",
        prompt_chars=len(prompt),
        response_chars=len(response_text),
    )
    return round_item


def select_candidates_with_legacy_sdk(
    model_name: str,
    prompt: str,
    audit_meta: dict[str, object] | None = None,
) -> CandidateSelection:
    if legacy_genai is None:
        raise RuntimeError("google-generativeai SDK is not installed")

    model = legacy_genai.GenerativeModel(model_name)
    started_at = perf_now()
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": GENERATION_TEMPERATURE,
                "top_p": GENERATION_TOP_P,
                "max_output_tokens": CANDIDATE_SELECTION_MAX_OUTPUT_TOKENS,
                "response_mime_type": "application/json",
            },
        )
    except Exception as exc:
        append_gemini_audit_log(
            "gemini.candidate_selection",
            model=model_name,
            sdk="legacy",
            prompt=prompt,
            response_text=None,
            status="sdk_error",
            error=str(exc),
            **(audit_meta or {}),
        )
        raise

    response_text = response.text or ""
    append_gemini_audit_log(
        "gemini.candidate_selection",
        model=model_name,
        sdk="legacy",
        prompt=prompt,
        response_text=response_text,
        status="ok" if response_text else "empty",
        **(audit_meta or {}),
    )
    payload = extract_json(response_text)
    if payload is None:
        raise RuntimeError(f"{model_name} returned non-JSON candidate-selection output")
    selection = CandidateSelection.model_validate(payload)
    log_perf(
        "genai.candidate_selection",
        started_at,
        model=model_name,
        sdk="legacy",
        prompt_chars=len(prompt),
        response_chars=len(response_text),
    )
    return selection


def generate_round_candidate(
    model_name: str,
    prompt: str,
    forbidden: list[str],
    audit_meta: dict[str, object] | None = None,
    locked_person_name: str | None = None,
    locked_actual_status: str | None = None,
) -> tuple[str, dict]:
    started_at = perf_now()
    try:
        if modern_genai is not None:
            round_item = generate_with_modern_sdk(model_name, prompt, audit_meta)
        else:
            round_item = generate_with_legacy_sdk(model_name, prompt, audit_meta)
        normalized = normalize_round(
            round_item,
            forbidden,
            locked_person_name=locked_person_name,
            locked_actual_status=locked_actual_status,
        )
    except Exception as exc:
        append_gemini_audit_log(
            "gemini.round_result",
            model=model_name,
            status="rejected",
            error=str(exc),
            **(audit_meta or {}),
        )
        raise

    append_gemini_audit_log(
        "gemini.round_result",
        model=model_name,
        status="accepted",
        person_name=normalized["person_name"],
        actual_status=normalized["actual_status"],
        **(audit_meta or {}),
    )
    log_perf(
        "round.candidate",
        started_at,
        model=model_name,
        person_name=normalized["person_name"],
        guessing_html_len=len(normalized["guessing_ui_html"]),
        reveal_html_len=len(normalized["reveal_ui_html"]),
    )
    return model_name, normalized


def choose_allowed_candidate(
    selection: CandidateSelection,
    forbidden: list[str],
) -> LockedCandidate:
    forbidden_keys = {normalize_name_key(name) for name in forbidden}
    seen_keys: set[str] = set()
    rejected_names: list[str] = []
    verification_errors: list[str] = []

    for candidate in selection.candidates:
        person_name = " ".join(candidate.person_name.split())
        key = normalize_name_key(person_name)
        if not person_name:
            continue
        if key in seen_keys or key in forbidden_keys:
            rejected_names.append(person_name)
            continue
        seen_keys.add(key)
        try:
            actual_status = determine_actual_status(person_name, candidate.actual_status)
        except Exception as exc:
            verification_errors.append(f"{person_name}: {exc}")
            continue
        return LockedCandidate(
            person_name=person_name,
            actual_status=actual_status,
            source="candidate_selection",
        )

    rejected_tail = ", ".join(rejected_names[:12]) if rejected_names else "none"
    verification_tail = (
        "; Unverified candidates: " + " | ".join(verification_errors[:5])
        if verification_errors
        else ""
    )
    raise ValueError(
        f"Candidate selection returned no allowed person. Rejected candidates: {rejected_tail}{verification_tail}"
    )


def choose_emergency_fallback_candidate(
    forbidden: list[str],
    source: str = "emergency_fallback",
) -> LockedCandidate:
    forbidden_order = {
        normalize_name_key(name): index for index, name in enumerate(forbidden)
    }
    fallback_index = {name: i for i, name in enumerate(EMERGENCY_FALLBACK_NAMES)}
    ranked_names = sorted(
        EMERGENCY_FALLBACK_NAMES,
        key=lambda name: (
            0 if normalize_name_key(name) not in forbidden_order else 1,
            forbidden_order.get(normalize_name_key(name), -1),
            fallback_index.get(name, len(EMERGENCY_FALLBACK_NAMES)),
        ),
    )
    verification_errors: list[str] = []
    for person_name in ranked_names:
        if normalize_name_key(person_name) in forbidden_order:
            continue
        try:
            actual_status = determine_actual_status(person_name, None)
        except Exception as exc:
            verification_errors.append(f"{person_name}: {exc}")
            continue
        return LockedCandidate(
            person_name=person_name,
            actual_status=actual_status,
            source=source,
        )

    verification_tail = (
        " Verification failures: " + " | ".join(verification_errors[:5])
        if verification_errors
        else ""
    )
    raise ValueError(
        "Emergency fallback has no unused, status-verified candidate."
        + verification_tail
    )


def should_use_emergency_fallback(error_message: str) -> bool:
    normalized = error_message.casefold()
    return (
        "candidate selection returned no allowed person" in normalized
        or "all candidate-selection models failed" in normalized
    )


def select_allowed_candidate_sync(
    forbidden: list[str],
    model_candidates: list[str],
    retry_notes: list[str],
    audit_meta: dict[str, object],
    category: str = "All Celebrities",
) -> tuple[str, LockedCandidate]:
    if (
        LOCAL_CANDIDATE_FIRST_HISTORY_SIZE > 0
        and len({normalize_name_key(name) for name in forbidden})
        >= LOCAL_CANDIDATE_FIRST_HISTORY_SIZE
    ):
        try:
            candidate = choose_emergency_fallback_candidate(
                forbidden,
                source="local_verified_candidate_pool",
            )
            append_gemini_audit_log(
                "gemini.candidate_selection_result",
                model="local-verified-candidate-pool",
                status="accepted",
                person_name=candidate.person_name,
                actual_status=candidate.actual_status,
                reason="history_size_threshold",
                forbidden_count=len(forbidden),
                **audit_meta,
            )
            return "local-verified-candidate-pool", candidate
        except Exception as exc:
            logger.warning("Local verified candidate selection failed: %s", exc)

    prompt = build_candidate_selection_prompt(forbidden, retry_notes, category)
    errors: list[str] = []

    for model_name in model_candidates:
        try:
            if modern_genai is not None:
                selection = select_candidates_with_modern_sdk(model_name, prompt, audit_meta)
            else:
                selection = select_candidates_with_legacy_sdk(model_name, prompt, audit_meta)
            candidate = choose_allowed_candidate(selection, forbidden)
            append_gemini_audit_log(
                "gemini.candidate_selection_result",
                model=model_name,
                status="accepted",
                person_name=candidate.person_name,
                actual_status=candidate.actual_status,
                **audit_meta,
            )
            return model_name, candidate
        except Exception as exc:
            append_gemini_audit_log(
                "gemini.candidate_selection_result",
                model=model_name,
                status="rejected",
                error=str(exc),
                **audit_meta,
            )
            logger.warning("Candidate selection failed for %s: %s", model_name, exc)
            errors.append(f"{model_name}: {exc}")

    raise RuntimeError("All candidate-selection models failed. " + " | ".join(errors))


def generate_single_round_sync(
    forbidden: list[str],
    model_candidates: list[str],
    race_models: bool = False,
    category: str = "All Celebrities",
) -> tuple[str, dict]:
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

    errors = []
    working_forbidden = list(dict.fromkeys(forbidden))
    retry_notes: list[str] = []

    for attempt in range(ROUND_GENERATION_ATTEMPTS):
        audit_request_id = str(uuid.uuid4())
        audit_meta = {
            "attempt": attempt + 1,
            "audit_request_id": audit_request_id,
        }
        locked_candidate: LockedCandidate | None = None
        try:
            _, locked_candidate = select_allowed_candidate_sync(
                working_forbidden,
                model_candidates,
                retry_notes,
                audit_meta,
                category=category,
            )
        except Exception as exc:
            errors.append(f"candidate selection attempt {attempt + 1}: {exc}")
            retry_notes.append(build_retry_note(str(exc)))
            retry_notes = list(dict.fromkeys(retry_notes))
            if should_use_emergency_fallback(str(exc)):
                try:
                    locked_candidate = choose_emergency_fallback_candidate(working_forbidden)
                except Exception as fallback_exc:
                    errors.append(
                        f"emergency fallback attempt {attempt + 1}: {fallback_exc}"
                    )
                    retry_notes.append(build_retry_note(str(fallback_exc)))
                    retry_notes = list(dict.fromkeys(retry_notes))
                    continue
                append_gemini_audit_log(
                    "gemini.candidate_selection_fallback",
                    status="used",
                    person_name=locked_candidate.person_name,
                    source=locked_candidate.source,
                    reason=str(exc),
                    **audit_meta,
                )
                logger.warning(
                    "Using emergency fallback candidate %s after candidate-selection failure on attempt %s: %s",
                    locked_candidate.person_name,
                    attempt + 1,
                    exc,
                )
            else:
                continue

        generation_forbidden = [
            name
            for name in working_forbidden
            if normalize_name_key(name) != normalize_name_key(locked_candidate.person_name)
        ]

        prompt = build_generation_prompt(
            generation_forbidden,
            retry_notes,
            locked_person_name=locked_candidate.person_name,
            locked_actual_status=locked_candidate.actual_status,
            category=category,
        )

        if race_models and len(model_candidates) > 1:
            forbidden_names_to_add = []
            attempt_retry_notes = []
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=len(model_candidates)
            )
            future_to_model = {
                executor.submit(
                    generate_round_candidate,
                    model_name,
                    prompt,
                    generation_forbidden,
                    audit_meta | {"race_mode": True},
                    locked_candidate.person_name,
                    locked_candidate.actual_status,
                ): model_name
                for model_name in model_candidates
            }
            executor_shutdown = False
            try:
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        for pending_future in future_to_model:
                            if pending_future is not future:
                                pending_future.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        executor_shutdown = True
                        return result
                    except Exception as exc:
                        logger.warning(
                            "Generation failed for %s on attempt %s: %s",
                            model_name,
                            attempt + 1,
                            exc,
                        )
                        errors.append(f"{model_name} attempt {attempt + 1}: {exc}")
                        attempt_retry_notes.append(build_retry_note(str(exc)))
                        if str(exc).startswith("Model returned forbidden person:"):
                            forbidden_names_to_add.append(str(exc).split(": ", 1)[1])
            finally:
                if not executor_shutdown:
                    executor.shutdown(wait=False, cancel_futures=True)

            working_forbidden.extend(forbidden_names_to_add)
            working_forbidden = list(dict.fromkeys(working_forbidden))
            retry_notes = list(dict.fromkeys(retry_notes + attempt_retry_notes))
            continue

        for model_name in model_candidates:
            try:
                return generate_round_candidate(
                    model_name,
                    prompt,
                    generation_forbidden,
                    audit_meta | {"race_mode": False},
                    locked_candidate.person_name,
                    locked_candidate.actual_status,
                )
            except Exception as exc:
                logger.warning(
                    "Generation failed for %s on attempt %s: %s",
                    model_name,
                    attempt + 1,
                    exc,
                )
                errors.append(f"{model_name} attempt {attempt + 1}: {exc}")
                if str(exc).startswith("Model returned forbidden person:"):
                    working_forbidden.append(str(exc).split(": ", 1)[1])
                retry_notes.append(build_retry_note(str(exc)))
                retry_notes = list(dict.fromkeys(retry_notes))

    raise RuntimeError("All configured Gemini models failed. " + " | ".join(errors))


def render_history_item(entry: dict) -> str:
    person = html.escape(str(entry.get("person", "Unknown")))
    status = html.escape(str(entry.get("status", ""))).upper()
    result_text = "Correct" if entry.get("correct") else "Missed"
    tone = "border-emerald-500/40 text-emerald-300" if entry.get("correct") else "border-rose-500/40 text-rose-300"
    return (
        f"<div class='rounded-2xl border {tone} bg-zinc-900/80 p-4'>"
        f"<p class='text-xs uppercase tracking-[0.35em] text-zinc-500'>Round</p>"
        f"<p class='mt-2 text-xl font-black text-white'>{person}</p>"
        f"<p class='mt-1 text-sm text-zinc-300'>Actual status: {status}</p>"
        f"<p class='mt-3 text-sm font-bold uppercase tracking-[0.2em]'>{result_text}</p>"
        f"</div>"
    )

def inject_round_number(html_str: str, round_num: int, mode: str) -> str:
    mode_display = str(mode).capitalize()
    injection = (
        f"<div class='pointer-events-none fixed left-4 top-4 z-[60] flex'>"
        f"<div class='rounded-full border border-white/20 bg-black/50 backdrop-blur-md px-5 py-2 shadow-2xl'>"
        f"<span class='text-sm font-bold uppercase tracking-widest text-white/90'>{mode_display} &bull; Round {round_num}</span>"
        f"</div>"
        f"</div>"
    )
    return injection + html_str



def render_finale_html(session: dict) -> str:
    score_val = int(session.get("score", 0))
    history_data = session.get("history", [])
    history_list = "".join(render_history_item(item) for item in history_data)

    return f"""
    <div class='w-full max-w-5xl rounded-[2.5rem] border border-white/10 bg-zinc-950/95 p-8 md:p-12 shadow-2xl'>
        <div class='text-center'>
            <p class='text-sm uppercase tracking-[0.5em] text-zinc-500'>Alive or Dead</p>
            <h1 class='mt-4 text-5xl md:text-7xl font-black tracking-tight uppercase'>Game Over</h1>
            <p class='mt-6 text-3xl md:text-5xl font-black text-emerald-400'>Final Score: {score_val} / {ROUNDS_PER_GAME}</p>
        </div>
        <div class='mt-10 grid gap-4 md:grid-cols-2'>
            {history_list}
        </div>
        <div class='mt-10 flex justify-center'>
            <button onclick='location.reload()' class='rounded-full bg-white px-10 py-4 text-lg font-black uppercase tracking-[0.2em] text-black transition hover:scale-105 active:scale-95'>
                Play Again
            </button>
        </div>
    </div>
    """


def collect_session_reserved_names(session: dict) -> list[str]:
    reserved_names: list[str] = []
    reserved_names.extend(session.get("used_names", []))
    active_round = session.get("active_round") or {}
    if active_round.get("person_name"):
        reserved_names.append(str(active_round["person_name"]))
    for round_data in session.get("queued_rounds", []):
        if isinstance(round_data, dict) and round_data.get("person_name"):
            reserved_names.append(str(round_data["person_name"]))
    for history_entry in session.get("history", []):
        if isinstance(history_entry, dict) and history_entry.get("person"):
            reserved_names.append(str(history_entry["person"]))
    return merge_names_recency_preserving([], reserved_names)


async def generate_round_payload(
    session: dict,
    target_round_number: int,
    model_candidates: list[str],
    race_models: bool = False,
) -> tuple[str, dict]:
    forbidden = collect_session_reserved_names(session)
    if session.get("mode", "survival") != "survival" and GLOBAL_HISTORY_PROMPT_LIMIT > 0:
        forbidden.extend(get_global_history()[-GLOBAL_HISTORY_PROMPT_LIMIT:])
    forbidden = merge_names_recency_preserving([], forbidden)
    model_name, round_data = await asyncio.to_thread(
        generate_single_round_sync,
        forbidden,
        model_candidates,
        race_models,
        category=session.get("category", "All Celebrities")
    )
    session["used_names"] = merge_names_recency_preserving(
        session.get("used_names", []),
        [round_data["person_name"]],
    )
    runtime_state["last_model"] = model_name
    runtime_state["last_error"] = None
    update_global_history([round_data["person_name"]])
    logger.info("Generated round %s using %s", target_round_number, model_name)
    return model_name, round_data


async def generate_round_for_session(
    session: dict,
    target_round_number: int,
    model_candidates: list[str],
    race_models: bool = False,
) -> dict:
    _, round_data = await generate_round_payload(
        session,
        target_round_number,
        model_candidates,
        race_models,
    )
    session["active_round"] = round_data
    return round_data


async def prefetch_next_rounds(session_id: str) -> None:
    session = sessions.get(session_id)
    if not session:
        return

    try:
        while True:
            current_session = sessions.get(session_id)
            if current_session is None:
                return
            if len(current_session["queued_rounds"]) >= PREFETCH_ROUND_BUFFER:
                return
            if current_session.get("mode") == "classic" and current_session["round_number"] + len(current_session["queued_rounds"]) >= ROUNDS_PER_GAME:
                return

            target_round_number = (
                current_session["round_number"] + len(current_session["queued_rounds"]) + 1
            )
            _, round_data = await generate_round_payload(
                current_session,
                target_round_number,
                BACKGROUND_MODEL_CANDIDATES,
                False,
            )
            current_session = sessions.get(session_id)
            if current_session is None:
                return
            round_lock: asyncio.Lock = current_session.get("round_lock") or asyncio.Lock()
            async with round_lock:
                current_session["queued_rounds"].append(round_data)
                current_session["prefetch_error"] = None
    except Exception as exc:
        current_session = sessions.get(session_id)
        if current_session is not None:
            current_session["prefetch_error"] = str(exc)
        logger.warning("Prefetch failed for session %s: %s", session_id, exc)
    finally:
        current_session = sessions.get(session_id)
        if current_session is not None:
            current_session["prefetch_task"] = None


def schedule_prefetch(session_id: str) -> None:
    session = sessions.get(session_id)
    if not session:
        return
    if session.get("mode") == "classic" and session["round_number"] >= ROUNDS_PER_GAME:
        return
    if len(session.get("queued_rounds", [])) >= PREFETCH_ROUND_BUFFER:
        return

    existing_task = session.get("prefetch_task")
    if existing_task is not None and not existing_task.done():
        return

    session["prefetch_task"] = asyncio.create_task(prefetch_next_rounds(session_id))


async def get_next_round_for_session(session_id: str, session: dict) -> dict:
    round_lock: asyncio.Lock = session.get("round_lock") or asyncio.Lock()

    # Fast path: grab from the prefetch queue while holding the lock.
    async with round_lock:
        queued_rounds = session.get("queued_rounds", [])
        if queued_rounds:
            queued_round = queued_rounds.pop(0)
            session["active_round"] = queued_round
            return queued_round

    # Wait for a running prefetch to deliver a round, then dequeue under the lock.
    existing_task = session.get("prefetch_task")
    if existing_task is not None and not existing_task.done():
        while True:
            async with round_lock:
                queued_rounds = session.get("queued_rounds", [])
                if queued_rounds:
                    queued_round = queued_rounds.pop(0)
                    session["active_round"] = queued_round
                    return queued_round

            existing_task = session.get("prefetch_task")
            if existing_task is None or existing_task.done():
                break

            await asyncio.sleep(0.05)

    return await generate_round_for_session(
        session,
        session["round_number"] + 1,
        FAST_MODEL_CANDIDATES,
        RACE_FAST_MODELS,
    )


async def _session_cleanup_loop() -> None:
    """Background coroutine that evicts sessions older than SESSION_TTL_SECONDS."""
    while True:
        await asyncio.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)
        cutoff = time.monotonic() - SESSION_TTL_SECONDS
        stale = [
            sid for sid, s in list(sessions.items())
            if s.get("created_at", 0) < cutoff
        ]
        for sid in stale:
            session = sessions.pop(sid, None)
            if session is None:
                continue
            task = session.get("prefetch_task")
            if task and not task.done():
                task.cancel()
        if stale:
            logger.info("Session cleanup: evicted %d stale session(s)", len(stale))




@asynccontextmanager
async def _lifespan(app):  # noqa: ARG001
    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Alive or Dead POC", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/static/{filename}")
async def serve_static(filename: str):
    path = Path(filename)
    if path.is_file() and filename.endswith(".png"):
        return FileResponse(path)
    raise HTTPException(status_code=404)

@app.get("/", response_class=HTMLResponse)
async def index():
    start_session_url = app_path("/api/start-session")
    next_round_url = app_path("/api/next-round")
    guess_url = app_path("/api/guess")
    leaderboard_url = app_path("/api/leaderboard")
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Alive or Dead? v{APP_VERSION}</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
            body {{
                font-family: 'Inter', sans-serif;
                background:
                    radial-gradient(circle at 20% 20%, rgba(251, 191, 36, 0.16), transparent 24%),
                    radial-gradient(circle at 82% 18%, rgba(244, 63, 94, 0.16), transparent 20%),
                    radial-gradient(circle at 50% 75%, rgba(59, 130, 246, 0.12), transparent 26%),
                    linear-gradient(180deg, #120d0a 0%, #050505 68%);
                color: white;
            }}
            body::before {{
                content: "";
                position: fixed;
                inset: 0;
                pointer-events: none;
                background-image:
                    linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
                background-size: 28px 28px;
                mask-image: linear-gradient(180deg, rgba(0,0,0,0.9), rgba(0,0,0,0.2));
            }}
        </style>
    </head>
    <body class="min-h-screen overflow-x-hidden">
        
        <div id="app" class="mx-auto flex min-h-screen w-full max-w-7xl items-center justify-center px-4 py-8">
            <div class="grid w-full items-center gap-10 lg:grid-cols-[1.1fr_0.9fr]">
                <div>
                    <p class="inline-flex rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-bold uppercase tracking-[0.35em] text-amber-200">
                        Real-Time Generative UI
                    </p>
                    <h1 class="mt-6 text-6xl font-black uppercase leading-[0.9] tracking-[-0.06em] sm:text-7xl lg:text-[8rem]">
                        Alive
                        <span class="block text-zinc-500">or</span>
                        Dead
                    </h1>
                    <p class="mt-6 max-w-2xl text-lg leading-relaxed text-zinc-300 sm:text-xl">
                        Experience the ultimate trivia showdown! Gemini AI dynamically crafts every single round just for you, digging up fascinating facts while our backend frantically scours Wikimedia to find the perfect portrait. Will you survive the AI's unpredictable challenges?
                    </p>
                    
                    <!-- New Game Options -->
                    <div class="mt-8 grid gap-4 max-w-md">
                        <div>
                            <label class="block text-xs uppercase tracking-[0.2em] text-zinc-400 mb-2">Game Mode</label>
                            <select id="mode-select" class="w-full rounded-xl border border-white/10 bg-zinc-900 px-4 py-3 text-white outline-none focus:border-amber-400">
                                <option value="survival" selected>Infinite Survival (10s Timer)</option>
                                <option value="classic">Classic (10 Rounds)</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-xs uppercase tracking-[0.2em] text-zinc-400 mb-2">Category</label>
                            <p class="text-sm text-zinc-500 mb-3">Choose a preselected category or enter your own below:</p>
                            <div class="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-3">
                                <button onclick="setCategory('All Celebrities')" class="cat-btn rounded-lg border border-white/10 bg-white/5 py-2 text-xs uppercase text-zinc-300 hover:bg-white/10">All Celebs</button>
                                <button onclick="setCategory('TV and Movies')" class="cat-btn rounded-lg border border-white/10 bg-white/5 py-2 text-xs uppercase text-zinc-300 hover:bg-white/10">TV & Movies</button>
                                <button onclick="setCategory('Rock & Roll Legends')" class="cat-btn rounded-lg border border-white/10 bg-white/5 py-2 text-xs uppercase text-zinc-300 hover:bg-white/10">Rockstars</button>
                                <button onclick="setCategory('Legendary Comedians')" class="cat-btn rounded-lg border border-white/10 bg-white/5 py-2 text-xs uppercase text-zinc-300 hover:bg-white/10">Comedians</button>
                                <button onclick="setCategory('Historical Figures')" class="cat-btn rounded-lg border border-white/10 bg-white/5 py-2 text-xs uppercase text-zinc-300 hover:bg-white/10">Historical</button>
                                <button onclick="setCategory('Sports Legends')" class="cat-btn rounded-lg border border-white/10 bg-white/5 py-2 text-xs uppercase text-zinc-300 hover:bg-white/10">Sports</button>
                            </div>
                            <input type="text" id="category-input" placeholder="Or type a custom category..." class="w-full rounded-xl border border-white/10 bg-zinc-900 px-4 py-3 text-white outline-none focus:border-amber-400" value="All Celebrities">
                        </div>
                    </div>

                    <div class="mt-8 flex flex-wrap items-center gap-4">
                        <button id="start-btn" onclick="startGame()" class="rounded-full bg-white px-12 py-5 text-lg font-black uppercase tracking-[0.24em] text-black transition hover:scale-105 active:scale-95">
                            Start Game
                        </button>
                        <button onclick="showLeaderboard()" class="rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm uppercase tracking-[0.28em] text-zinc-300 hover:bg-white/10">
                            Leaderboard
                        </button>
                    </div>
                </div>
                <div class="relative hidden lg:block">
                    <div class="absolute inset-0 rotate-[-7deg] rounded-[2.75rem] bg-gradient-to-br from-amber-400/35 via-rose-500/20 to-sky-400/20 blur-3xl"></div>
                    <img src="{app_path('/static/gameplay_real.png')}" class="relative w-full h-auto object-cover rounded-[2.5rem] border border-white/10 shadow-2xl" alt="Gameplay screenshot">
                </div>
            </div>
        </div>

        <!-- Leaderboard Modal -->
        <div id="leaderboard-modal" class="fixed inset-0 z-50 hidden items-center justify-center bg-black/80 backdrop-blur-sm p-4">
            <div class="w-full max-w-lg rounded-3xl border border-white/10 bg-zinc-900 p-8 shadow-2xl">
                <h2 id="lb-title" class="text-3xl font-black uppercase text-white mb-6 text-center">Global Leaderboard</h2>
                
                <div id="lb-submit-section" class="hidden mb-8 border-b border-white/10 pb-8 text-center">
                    <p class="text-rose-400 font-bold uppercase tracking-widest text-sm mb-2">Game Over</p>
                    <p class="text-5xl font-black text-amber-300 mb-4" id="lb-score-display">12</p>
                    <p class="text-zinc-400 mb-4">Enter 3 initials to save your survival streak!</p>
                    <div class="flex justify-center gap-2">
                        <input type="text" id="lb-initials" maxlength="3" class="w-24 text-center rounded-lg border border-white/20 bg-black py-3 text-2xl font-black text-white uppercase outline-none focus:border-amber-400">
                        <button onclick="submitScore()" class="rounded-lg bg-amber-500 px-6 font-bold uppercase text-black hover:bg-amber-400">Submit</button>
                    </div>
                </div>

                <div id="lb-list" class="flex flex-col gap-3">
                    <p class="text-center text-zinc-500">Loading scores...</p>
                </div>

                <div class="mt-8 text-center">
                    <button onclick="closeLeaderboard()" class="rounded-full border border-white/20 px-8 py-3 text-sm font-bold uppercase tracking-widest text-white hover:bg-white/10">Close</button>
                </div>
            </div>
        </div>

        <script>

            let sessionId = null;
            let sessionMode = 'survival';
            let survivalTimer = null;
            let isGameOver = false;
            const START_SESSION_URL = {json.dumps(start_session_url)};
            const NEXT_ROUND_URL = {json.dumps(next_round_url)};
            const GUESS_URL = {json.dumps(guess_url)};
            const LEADERBOARD_URL = {json.dumps(leaderboard_url)};

            function setCategory(cat) {{
                document.getElementById('category-input').value = cat;
            }}

            function startSurvivalTimer() {{
                clearTimeout(survivalTimer);
                const oldBar = document.getElementById('survival-timer-bar');
                if(oldBar) oldBar.remove();

                if (sessionMode !== 'survival') return;

                const barHtml = `
                <div id="survival-timer-bar" style="position:fixed; top:0; left:0; height:8px; background:linear-gradient(90deg, #f59e0b, #ef4444); width:100%; transition: width 10s linear; z-index:9999;">
                    <div style="position:absolute; right:10px; top:12px; color:#ef4444; font-weight:bold; font-size:1.5rem; text-transform:uppercase; animation: pulse 1s infinite;">Hurry! 10 Seconds!</div>
                </div>`;
                document.body.insertAdjacentHTML('beforeend', barHtml);
                
                setTimeout(() => {{
                    const bar = document.getElementById('survival-timer-bar');
                    if(bar) bar.style.width = '0%';
                }}, 50);

                survivalTimer = setTimeout(() => {{
                    submitGuess('timeout');
                }}, 10000);
            }}

            function stopSurvivalTimer() {{
                clearTimeout(survivalTimer);
                const bar = document.getElementById('survival-timer-bar');
                if(bar) bar.remove();
            }}

            async function showLeaderboard(finalScore = null) {{
                document.getElementById('leaderboard-modal').classList.remove('hidden');
                document.getElementById('leaderboard-modal').classList.add('flex');
                
                const submitSection = document.getElementById('lb-submit-section');
                if (finalScore !== null) {{
                    submitSection.classList.remove('hidden');
                    document.getElementById('lb-score-display').textContent = finalScore;
                }} else {{
                    submitSection.classList.add('hidden');
                }}

                try {{
                    const res = await fetch(LEADERBOARD_URL);
                    const scores = await res.json();
                    const list = document.getElementById('lb-list');
                    list.innerHTML = scores.map((s, i) => `
                        <div class="flex justify-between items-center bg-white/5 p-3 rounded-lg border border-white/5">
                            <span class="font-black text-amber-500 w-8">#${{i+1}}</span>
                            <span class="font-bold text-white text-xl tracking-widest">${{s.initials}}</span>
                            <span class="font-black text-rose-400 text-xl">${{s.score}}</span>
                        </div>
                    `).join('');
                }} catch(e) {{}}
            }}

            function closeLeaderboard() {{
                document.getElementById('leaderboard-modal').classList.add('hidden');
                document.getElementById('leaderboard-modal').classList.remove('flex');
                if (typeof isGameOver !== 'undefined' && isGameOver) {{
                    location.reload();
                }} else if (!document.getElementById('lb-submit-section').classList.contains('hidden')) {{
                    location.reload();
                }}
            }}

            async function submitScore() {{
                const initials = document.getElementById('lb-initials').value.toUpperCase() || 'ANON';
                const score = parseInt(document.getElementById('lb-score-display').textContent);
                await fetch(LEADERBOARD_URL, {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{ initials, score }})
                }});
                document.getElementById('lb-submit-section').classList.add('hidden');
                showLeaderboard();
            }}

            function renderLoader(title, detail) {{
                stopSurvivalTimer();
                document.getElementById('app').innerHTML = `
                    <div class="relative w-full max-w-4xl overflow-hidden rounded-[2.75rem] border border-white/10 bg-zinc-950/95 p-10 text-center shadow-2xl">
                        <div class="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-amber-300 via-rose-400 to-sky-400"></div>
                        <div class="mx-auto mb-8 flex h-24 w-24 items-center justify-center rounded-full border border-white/15 bg-white/5">
                            <div class="h-16 w-16 rounded-full border-[6px] border-white border-t-transparent animate-spin"></div>
                        </div>
                        <p class="text-xs font-bold uppercase tracking-[0.42em] text-zinc-500">Live generation</p>
                        <h2 id="status-title" class="mt-4 text-3xl font-black uppercase tracking-[0.18em] md:text-4xl"></h2>
                        <p id="status-detail" class="mx-auto mt-5 max-w-2xl text-lg leading-relaxed text-zinc-400"></p>
                        <div id="loader-warning" class="mt-8 text-xl font-bold text-rose-500 uppercase tracking-widest animate-pulse hidden">
                            ⚠️ GET READY: 10 SECONDS PER ROUND! ⚠️
                        </div>
                    </div>`;
                document.getElementById('status-title').textContent = title;
                document.getElementById('status-detail').textContent = detail;
                
                if (typeof sessionMode !== 'undefined' && sessionMode === 'survival') {{
                    document.getElementById('loader-warning').classList.remove('hidden');
                }}
            }}

            function renderFatal(message) {{
                stopSurvivalTimer();
                document.getElementById('app').innerHTML = `
                    <div class="w-full max-w-3xl rounded-[2.75rem] border border-rose-500/30 bg-zinc-950/95 p-10 text-center shadow-2xl">
                        <p class="text-sm uppercase tracking-[0.45em] text-rose-300">Request Failed</p>
                        <h2 class="mt-4 text-4xl font-black uppercase tracking-tight">The Game Could Not Continue</h2>
                        <p id="fatal-message" class="mx-auto mt-5 max-w-2xl text-lg leading-relaxed text-zinc-300"></p>
                        <button onclick="location.reload()" class="mt-8 rounded-full bg-white px-8 py-4 text-sm font-black uppercase tracking-[0.2em] text-black transition hover:scale-105 active:scale-95">
                            Reload
                        </button>
                    </div>`;
                document.getElementById('fatal-message').textContent = message;
            }}

            async function startGame() {{
                sessionMode = document.getElementById('mode-select') ? document.getElementById('mode-select').value : 'survival';
                const category = document.getElementById('category-input') ? document.getElementById('category-input').value : 'All Celebrities';

                renderLoader(
                    'Designing Round One',
                    'Gemini is generating the first challenge in real time.'
                );

                try {{
                    const response = await fetch(START_SESSION_URL, {{ 
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ mode: sessionMode, category: category }})
                    }});
                    const data = await response.json();
                    if (!response.ok) {{
                        renderFatal(data.detail || 'Unable to start a new session.');
                        return;
                    }}
                    sessionId = data.session_id;
                    document.getElementById('app').innerHTML = data.guessing_ui_html;
                    startSurvivalTimer();
                }} catch (error) {{
                    renderFatal('Unable to start a new session.');
                }}
            }}

            async function loadNextRound() {{
                renderLoader(
                    'Designing Next Round',
                    'Gemini is generating the next challenge right now.'
                );

                try {{
                    const response = await fetch(`${{NEXT_ROUND_URL}}?session_id=${{encodeURIComponent(sessionId)}}`);
                    const data = await response.json();
                    if (!response.ok) {{
                        renderFatal(data.detail || 'Unable to load the next round.');
                        return;
                    }}
                    if (data.status === 'finale') {{
                        document.getElementById('app').innerHTML = data.finale_html;
                        return;
                    }}
                    document.getElementById('app').innerHTML = data.guessing_ui_html;
                    startSurvivalTimer();
                }} catch (error) {{
                    renderFatal('The next round failed to load.');
                }}
            }}

            async function submitGuess(guess) {{
                stopSurvivalTimer();
                try {{
                    const response = await fetch(GUESS_URL, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ session_id: sessionId, guess: guess }})
                    }});
                    const data = await response.json();
                    if (!response.ok) {{
                        renderFatal(data.detail || 'The guess request failed.');
                        return;
                    }}
                    document.getElementById('app').innerHTML = data.reveal_ui_html;

                    if (data.game_over && sessionMode === 'survival') {{
                        isGameOver = true;
                        const nextBtn = document.querySelector('button[onclick*="showLeaderboard"], button[onclick*="loadNextRound"]');
                        if (nextBtn) nextBtn.style.display = 'none';
                        // Disable the loadNextRound button injected by Gemini
                        setTimeout(() => {{
                            showLeaderboard(data.score);
                        }}, 2500);
                    }}
                }} catch (error) {{
                    renderFatal('The guess request failed.');
                }}
            }}
        </script>
    </body>
    </html>
    """


@app.get("/api/status")
async def status():
    return {
        "last_error": runtime_state["last_error"],
        "last_model": runtime_state["last_model"],
        "gemini_audit_log_enabled": GEMINI_AUDIT_LOG_ENABLED,
        "gemini_audit_log_file": str(GEMINI_AUDIT_LOG_FILE),
        "image_url_validation_enabled": IMAGE_URL_VALIDATION_ENABLED,
        "status_verification_enabled": STATUS_VERIFICATION_ENABLED,
        "status_verification_required": STATUS_VERIFICATION_REQUIRED,
        "local_candidate_first_history_size": LOCAL_CANDIDATE_FIRST_HISTORY_SIZE,
        "wikimedia_search_timeout_seconds": WIKIMEDIA_SEARCH_TIMEOUT_SECONDS,
        "image_contract": "backend_wikimedia_query_resolution",
    }


@app.post("/api/start-session")
async def start_session(request: FastAPIRequest):
    try:
        data = await request.json()
    except Exception:
        data = {}
    
    session_id = str(uuid.uuid4())
    session = {
        "score": 0,
        "round_number": 0,
        "history": [],
        "used_names": [],
        "active_round": None,
        "queued_rounds": [],
        "prefetch_task": None,
        "prefetch_error": None,
        "created_at": time.monotonic(),
        "round_lock": asyncio.Lock(),
        "mode": data.get("mode", "survival"),
        "category": data.get("category", "All Celebrities"),
    }
    sessions[session_id] = session

    try:
        round_data = await generate_round_for_session(
            session,
            1,
            FAST_MODEL_CANDIDATES,
            RACE_FAST_MODELS,
        )
    except Exception as exc:
        runtime_state["last_error"] = str(exc)
        sessions.pop(session_id, None)
        logger.error("start_session generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Round generation failed. Please try again.",
        ) from exc

    session["round_number"] = 1
    schedule_prefetch(session_id)
    return {
        "session_id": session_id,
        "status": "ready",
        "guessing_ui_html": inject_round_number(
            round_data["guessing_ui_html"], 
            session["round_number"], 
            session.get("mode", "survival")
        ),
    }


@app.get("/api/next-round")
async def next_round(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    if session.get("mode") == "classic" and session["round_number"] >= ROUNDS_PER_GAME:
        return {"status": "finale", "finale_html": render_finale_html(session)}
        
    if session.get("mode") == "survival" and session["history"] and not session["history"][-1]["correct"]:
        raise HTTPException(status_code=400, detail="Game Over. You cannot continue in survival mode after an incorrect guess.")

    try:
        round_data = await get_next_round_for_session(session_id, session)
    except Exception as exc:
        runtime_state["last_error"] = str(exc)
        logger.error("next_round generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Round generation failed. Please try again.",
        ) from exc

    session["round_number"] += 1
    schedule_prefetch(session_id)
    return {
        "status": "playing", 
        "guessing_ui_html": inject_round_number(
            round_data["guessing_ui_html"], 
            session["round_number"], 
            session.get("mode", "survival")
        )
    }



@app.get("/api/leaderboard")
async def get_leaderboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT initials, score, timestamp FROM leaderboard ORDER BY score DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    return [{"initials": r[0], "score": r[1], "date": r[2]} for r in rows]

class LeaderboardSubmit(BaseModel):
    initials: str
    score: int

@app.post("/api/leaderboard")
async def post_leaderboard(submission: LeaderboardSubmit):
    if len(submission.initials) > 10:
        submission.initials = submission.initials[:10]
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO leaderboard (initials, score) VALUES (?, ?)", (submission.initials, submission.score))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/api/guess")
async def guess(request: FastAPIRequest):
    data = await request.json()
    session_id = data.get("session_id")
    user_guess = str(data.get("guess", "")).lower()

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if user_guess not in {"alive", "dead", "timeout"}:
        raise HTTPException(status_code=400, detail="Guess must be 'alive', 'dead', or 'timeout'")

    session = sessions[session_id]
    round_data = session.get("active_round")
    if not round_data:
        raise HTTPException(status_code=400, detail="No active round to score")

    is_correct = user_guess == round_data["actual_status"].lower()
    if is_correct:
        session["score"] += 1

    session["history"].append(
        {
            "person": round_data["person_name"],
            "status": round_data["actual_status"],
            "correct": is_correct,
        }
    )

    if user_guess == "timeout":
        color = "text-amber-500"
        text = "OUT OF TIME"
    else:
        color = "text-emerald-400" if is_correct else "text-rose-400"
        text = "CORRECT" if is_correct else "INCORRECT"
        
    injection = (
        "<div class='pointer-events-none fixed left-0 right-0 top-8 z-50 flex justify-center'>"
        "<div class='rounded-full border border-white/10 bg-black/90 px-8 py-3 shadow-2xl'>"
        f"<span class='{color} text-2xl font-black uppercase tracking-[0.25em]'>{text}</span>"
        "</div>"
        "</div>"
    )
    
    game_over = False
    if session.get("mode") == "survival" and not is_correct:
        game_over = True
        logger.info(f"Game over survival: is_correct={is_correct}")
    reveal_html = injection + round_data["reveal_ui_html"]
    reveal_html = inject_round_number(
        reveal_html, 
        session.get("round_number", 0), 
        session.get("mode", "survival")
    )
    if game_over:
        reveal_html = reveal_html.replace('onclick="loadNextRound()"', 'onclick="showLeaderboard(' + str(session['score']) + ')"')
        reveal_html = reveal_html.replace("loadNextRound()", "showLeaderboard(" + str(session['score']) + ")")
        
    return {
        "reveal_ui_html": reveal_html,
        "game_over": game_over,
        "score": session["score"],
        "correct": is_correct
    }


if __name__ == "__main__":
    import uvicorn

    _host = os.getenv("UVICORN_HOST", "127.0.0.1")
    _port = int(os.getenv("UVICORN_PORT", "8000"))
    uvicorn.run(app, host=_host, port=_port)
