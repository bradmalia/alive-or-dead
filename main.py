import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
from html.parser import HTMLParser
import json
import logging
import os
import re
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request as UrlRequest, urlopen
from urllib.error import HTTPError

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from pydantic import field_validator, model_validator

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
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if legacy_genai and api_key:
    legacy_genai.configure(api_key=api_key)

BASE_DIR = Path(__file__).resolve().parent
IP_HISTORY_FILE = BASE_DIR / "ip_history.json"
CANDIDATE_IP_POOL_FILE = Path(
    os.getenv("CANDIDATE_IP_POOL_FILE", str(BASE_DIR / "ip_candidate_pools.json"))
).expanduser()
if not CANDIDATE_IP_POOL_FILE.is_absolute():
    CANDIDATE_IP_POOL_FILE = (BASE_DIR / CANDIDATE_IP_POOL_FILE).resolve()
CANDIDATE_BANK_FILE = Path(
    os.getenv("CANDIDATE_BANK_FILE", str(BASE_DIR / "candidate_bank.json"))
).expanduser()
if not CANDIDATE_BANK_FILE.is_absolute():
    CANDIDATE_BANK_FILE = (BASE_DIR / CANDIDATE_BANK_FILE).resolve()
PORTRAIT_RESOLUTION_CACHE_FILE = Path(
    os.getenv(
        "PORTRAIT_RESOLUTION_CACHE_FILE",
        str(BASE_DIR / "portrait_resolution_cache.json"),
    )
).expanduser()
if not PORTRAIT_RESOLUTION_CACHE_FILE.is_absolute():
    PORTRAIT_RESOLUTION_CACHE_FILE = (BASE_DIR / PORTRAIT_RESOLUTION_CACHE_FILE).resolve()


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

APP_VERSION = "3.6.2"
APP_BASE_PATH = normalize_app_base_path(os.getenv("APP_BASE_PATH", ""))
ROUNDS_PER_GAME = 10
ROUND_GENERATION_ATTEMPTS = int(os.getenv("ROUND_GENERATION_ATTEMPTS", "3"))
PREFETCH_ROUND_BUFFER = int(os.getenv("PREFETCH_ROUND_BUFFER", "2"))
CANDIDATE_SELECTION_COUNT = int(os.getenv("CANDIDATE_SELECTION_COUNT", "60"))
CANDIDATE_SELECTION_MIN_COUNT = int(os.getenv("CANDIDATE_SELECTION_MIN_COUNT", "24"))
SESSION_APPROVED_POOL_SIZE = int(
    os.getenv("SESSION_APPROVED_POOL_SIZE", str(ROUNDS_PER_GAME))
)
SESSION_CANDIDATE_TARGET_COUNT = int(
    os.getenv("SESSION_CANDIDATE_TARGET_COUNT", "30")
)
IP_CANDIDATE_POOL_MAX_ENTRIES = int(
    os.getenv("IP_CANDIDATE_POOL_MAX_ENTRIES", str(SESSION_CANDIDATE_TARGET_COUNT))
)
IP_CANDIDATE_POOL_TTL_SECONDS = int(
    os.getenv("IP_CANDIDATE_POOL_TTL_SECONDS", str(7 * 24 * 60 * 60))
)
CANDIDATE_BANK_MAX_ENTRIES = int(os.getenv("CANDIDATE_BANK_MAX_ENTRIES", "400"))
CANDIDATE_BANK_TTL_SECONDS = int(
    os.getenv("CANDIDATE_BANK_TTL_SECONDS", str(7 * 24 * 60 * 60))
)
IP_HISTORY_MAX_NAMES = int(os.getenv("IP_HISTORY_MAX_NAMES", "1000"))
SESSION_NEGOTIATION_ATTEMPTS = int(os.getenv("SESSION_NEGOTIATION_ATTEMPTS", "3"))
ROUND_UI_TEMPERATURE = float(os.getenv("ROUND_UI_TEMPERATURE", "0.65"))
ROUND_UI_TOP_P = float(os.getenv("ROUND_UI_TOP_P", "0.95"))
ROUND_UI_MAX_OUTPUT_TOKENS = int(os.getenv("ROUND_UI_MAX_OUTPUT_TOKENS", "3200"))
CANDIDATE_SELECTION_TEMPERATURE = float(
    os.getenv("CANDIDATE_SELECTION_TEMPERATURE", "0.35")
)
CANDIDATE_SELECTION_TOP_P = float(
    os.getenv("CANDIDATE_SELECTION_TOP_P", "0.8")
)
CANDIDATE_SELECTION_MAX_OUTPUT_TOKENS = int(
    os.getenv("CANDIDATE_SELECTION_MAX_OUTPUT_TOKENS", "4500")
)
RACE_FAST_MODELS = os.getenv("RACE_FAST_MODELS", "true").lower() == "true"
PERF_LOG_ENABLED = os.getenv("PERF_LOG_ENABLED", "true").lower() == "true"
GEMINI_AUDIT_LOG_ENABLED = os.getenv("GEMINI_AUDIT_LOG_ENABLED", "true").lower() == "true"
IMAGE_URL_VALIDATION_ENABLED = os.getenv("IMAGE_URL_VALIDATION_ENABLED", "true").lower() == "true"
STATUS_VALIDATION_ENABLED = os.getenv("STATUS_VALIDATION_ENABLED", "false").lower() == "true"
PORTRAIT_FALLBACK_MODE = os.getenv("PORTRAIT_FALLBACK_MODE", "prefer_real").strip().lower()
IMAGE_URL_VALIDATION_TIMEOUT_SECONDS = float(
    os.getenv("IMAGE_URL_VALIDATION_TIMEOUT_SECONDS", "4.0")
)
WIKIMEDIA_429_RETRY_COUNT = int(os.getenv("WIKIMEDIA_429_RETRY_COUNT", "2"))
WIKIMEDIA_429_RETRY_BACKOFF_MS = float(
    os.getenv("WIKIMEDIA_429_RETRY_BACKOFF_MS", "350")
)
WIKIMEDIA_MIN_REQUEST_INTERVAL_MS = float(
    os.getenv("WIKIMEDIA_MIN_REQUEST_INTERVAL_MS", "200")
)
PORTRAIT_RESOLUTION_CACHE_MAX_ENTRIES = int(
    os.getenv("PORTRAIT_RESOLUTION_CACHE_MAX_ENTRIES", "2000")
)
WIKIMEDIA_SEARCH_TIMEOUT_SECONDS = float(
    os.getenv("WIKIMEDIA_SEARCH_TIMEOUT_SECONDS", "3.0")
)
WIKIMEDIA_SEARCH_LIMIT = int(os.getenv("WIKIMEDIA_SEARCH_LIMIT", "8"))
APP_CONTACT_URL = os.getenv(
    "APP_CONTACT_URL",
    "https://www.maliaplace.com/alive-or-dead",
)
IMAGE_FETCH_USER_AGENT = os.getenv(
    "IMAGE_FETCH_USER_AGENT",
    f"AliveOrDeadPOC/{APP_VERSION} ({APP_CONTACT_URL}; image validation)",
)
WIKIMEDIA_API_USER_AGENT = os.getenv(
    "WIKIMEDIA_API_USER_AGENT",
    f"AliveOrDeadPOC/{APP_VERSION} ({APP_CONTACT_URL}; wikimedia search)",
)
FAST_MODEL_CANDIDATES = [
    item.strip()
    for item in os.getenv(
        "GEMINI_FAST_MODELS",
        "gemini-2.5-flash-lite,gemini-2.5-flash",
    ).split(",")
    if item.strip()
]
BACKGROUND_MODEL_CANDIDATES = [
    item.strip()
    for item in os.getenv(
        "GEMINI_BACKGROUND_MODELS",
        "gemini-2.5-flash,gemini-2.5-flash-lite",
    ).split(",")
    if item.strip()
]
GEMINI_AUDIT_LOG_FILE = Path(
    os.getenv("GEMINI_AUDIT_LOG_FILE", str(BASE_DIR / "gemini_audit.jsonl"))
).expanduser()
if not GEMINI_AUDIT_LOG_FILE.is_absolute():
    GEMINI_AUDIT_LOG_FILE = (BASE_DIR / GEMINI_AUDIT_LOG_FILE).resolve()

if PORTRAIT_FALLBACK_MODE not in {"fast", "prefer_real", "require_real"}:
    PORTRAIT_FALLBACK_MODE = "prefer_real"

GUESS_ALIVE_RE = re.compile(r"submitGuess\((['\"])alive\1\)")
GUESS_DEAD_RE = re.compile(r"submitGuess\((['\"])dead\1\)")
NEXT_ROUND_RE = re.compile(r"loadNextRound\(\)")
IMG_SRC_RE = re.compile(r"<img\b[^>]*\bsrc\s*=\s*(['\"])(.*?)\1", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
TAILWIND_OPACITY_RE = re.compile(r"opacity-(\d{1,3})$")
TAILWIND_SCALE_RE = re.compile(r"scale-(\d{2,3})$")
INLINE_OPACITY_RE = re.compile(r"opacity\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)", re.IGNORECASE)
INLINE_BLUR_RE = re.compile(r"blur\s*\(", re.IGNORECASE)
INLINE_SCALE_RE = re.compile(r"scale\s*\(\s*([0-9.]+)", re.IGNORECASE)
QUOTED_TEXT_RE = re.compile(r'"[^"]{1,160}"|\'[^\']{1,160}\'|“[^”]{1,160}”|‘[^’]{1,160}’')
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REMOTE_IMAGE_PREFIXES = ("http://", "https://", "//")
WIKIMEDIA_DOMAIN_MARKERS = ("wikimedia.org", "wikipedia.org")
PORTRAIT_IMAGE_PLACEHOLDER = "__PORTRAIT_IMAGE_URL__"
WIKIMEDIA_COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
WIKIPEDIA_PAGE_SEARCH_API_URL = "https://api.wikimedia.org/core/v1/wikipedia/en/search/page"
WIKIPEDIA_PAGE_SUMMARY_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKIPEDIA_ACTION_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIDATA_ENTITY_DATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/"
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
    "330",
    "500",
    "960",
    "1280",
    "1920",
    "3840",
}
SUPPORTED_RASTER_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
COMMONS_QUERY_NOISE_RE = re.compile(r"\b(?:wikimedia|wikipedia|commons)\b", re.IGNORECASE)
GENERIC_PORTRAIT_QUERY_TOKENS = {
    "portrait",
    "headshot",
    "publicity",
    "press",
    "photo",
    "photos",
    "publicityphoto",
}
CONTEXT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "their",
    "they",
    "them",
    "his",
    "her",
    "its",
    "into",
    "across",
    "known",
    "most",
    "famous",
    "fact",
    "once",
    "like",
    "both",
    "beloved",
    "iconic",
    "legendary",
    "career",
    "continues",
    "continued",
}
CONTEXT_FAMILY_KEYWORDS = {
    "acting": {
        "actor",
        "actress",
        "film",
        "films",
        "movie",
        "movies",
        "screen",
        "television",
        "tv",
        "sitcom",
        "series",
        "comedy",
        "comedian",
        "role",
        "roles",
        "blockbuster",
        "heroic",
        "hero",
        "star",
    },
    "music": {
        "singer",
        "song",
        "songs",
        "album",
        "albums",
        "music",
        "musician",
        "band",
        "guitar",
        "guitarist",
        "rapper",
        "composer",
        "performer",
    },
    "sports": {
        "athlete",
        "sports",
        "player",
        "football",
        "baseball",
        "basketball",
        "tennis",
        "boxing",
        "boxer",
        "olympic",
        "golfer",
        "racer",
    },
    "politics": {
        "president",
        "politician",
        "senator",
        "governor",
        "campaign",
        "white",
        "house",
        "statesman",
        "minister",
        "prime",
    },
    "science": {
        "scientist",
        "physicist",
        "astronaut",
        "inventor",
        "mathematician",
        "researcher",
        "professor",
    },
}
PAGE_ROLE_MISMATCH_KEYWORDS = {
    "acting": {"presenter", "broadcaster", "radio", "host", "dj"},
}
GUESSING_PORTRAIT_MIN_OPACITY = float(
    os.getenv("GUESSING_PORTRAIT_MIN_OPACITY", "0.75")
)
GUESSING_PORTRAIT_MAX_SCALE = float(
    os.getenv("GUESSING_PORTRAIT_MAX_SCALE", "1.1")
)
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
NON_PERSON_PAGE_KEYWORDS = {
    "aircraft",
    "airplane",
    "cargo",
    "boat",
    "bus",
    "car",
    "cruise",
    "destroyer",
    "ferry",
    "helicopter",
    "liner",
    "locomotive",
    "ocean",
    "plane",
    "ship",
    "sail",
    "submarine",
    "tank",
    "tanker",
    "tram",
    "train",
    "vehicle",
    "vessel",
    "warship",
    "yacht",
    "rms",
    "hms",
    "uss",
    "ss",
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
    ]
]
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
    date_of_birth: str = Field(min_length=10, max_length=10)
    date_of_death: str | None = Field(default=None, min_length=10, max_length=10)
    portrait_search_query: str = Field(min_length=3, max_length=200)
    next_round_label: str = Field(default="Next Round", max_length=80)
    guessing_ui_html: str = Field(min_length=50, max_length=5000)
    reveal_ui_html: str = Field(min_length=50, max_length=5000)

    @field_validator("actual_status", mode="before")
    @classmethod
    def normalize_actual_status(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "deceased":
                return "dead"
            return normalized
        return value

    @field_validator("date_of_birth")
    @classmethod
    def validate_date_of_birth(cls, value: str) -> str:
        normalized = value.strip()
        if not ISO_DATE_RE.fullmatch(normalized):
            raise ValueError("date_of_birth must use YYYY-MM-DD format")
        datetime.strptime(normalized, "%Y-%m-%d")
        return normalized

    @field_validator("date_of_death", mode="before")
    @classmethod
    def normalize_date_of_death(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("date_of_death")
    @classmethod
    def validate_date_of_death(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not ISO_DATE_RE.fullmatch(value):
            raise ValueError("date_of_death must use YYYY-MM-DD format")
        datetime.strptime(value, "%Y-%m-%d")
        return value

    @field_validator("next_round_label", mode="before")
    @classmethod
    def normalize_next_round_label(cls, value):
        if isinstance(value, str):
            return " ".join(value.split())
        return value

    @model_validator(mode="after")
    def validate_status_dates(self):
        if self.actual_status == "alive" and self.date_of_death is not None:
            raise ValueError("actual_status/date_of_death mismatch: alive people must not include date_of_death")
        if self.actual_status == "dead" and self.date_of_death is None:
            raise ValueError("actual_status/date_of_death mismatch: dead people must include date_of_death")
        if self.date_of_death and self.date_of_birth > self.date_of_death:
            raise ValueError("date_of_birth must not be later than date_of_death")
        return self


class CandidatePerson(BaseModel):
    person_name: str = Field(min_length=1, max_length=100)
    actual_status: Literal["alive", "dead"]

    @field_validator("actual_status", mode="before")
    @classmethod
    def normalize_actual_status(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "deceased":
                return "dead"
            return normalized
        return value


class CandidateSelection(BaseModel):
    candidates: list[CandidatePerson] = Field(min_length=1, max_length=200)


@dataclass(frozen=True)
class LockedCandidate:
    person_name: str
    actual_status: str | None = None
    source: str = "candidate_selection"


class CandidatePortraitUnavailableError(RuntimeError):
    pass


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
image_url_validation_cache: dict[str, tuple[bool, str]] = {}
image_url_validation_lock = threading.Lock()
portrait_resolution_cache: dict[tuple[str, str], tuple[bool, str]] = {}
portrait_resolution_lock = threading.Lock()
status_resolution_cache: dict[str, str | None] = {}
status_resolution_lock = threading.Lock()
ip_history_lock = threading.Lock()
candidate_ip_pool_lock = threading.Lock()
candidate_bank_lock = threading.Lock()
wikimedia_request_lock = threading.Lock()
last_wikimedia_request_at = 0.0


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
            with GEMINI_AUDIT_LOG_FILE.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:
        logger.warning("Failed to write Gemini audit log: %s", exc)


def persist_portrait_resolution_cache() -> None:
    try:
        records: list[dict[str, str]] = []
        with portrait_resolution_lock:
            for cache_key, cached_value in portrait_resolution_cache.items():
                ok, image_source = cached_value
                if not ok or not str(image_source).lower().startswith(("http://", "https://")):
                    continue
                person_name_key, portrait_query_key, profession_hint = cache_key
                records.append(
                    {
                        "person_name_key": person_name_key,
                        "portrait_query_key": portrait_query_key,
                        "profession_hint": profession_hint or "",
                        "image_source": image_source,
                    }
                )
        if PORTRAIT_RESOLUTION_CACHE_MAX_ENTRIES > 0:
            records = records[-PORTRAIT_RESOLUTION_CACHE_MAX_ENTRIES:]
        PORTRAIT_RESOLUTION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp_path = PORTRAIT_RESOLUTION_CACHE_FILE.with_suffix(
            PORTRAIT_RESOLUTION_CACHE_FILE.suffix + ".tmp"
        )
        temp_path.write_text(
            json.dumps(records, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(PORTRAIT_RESOLUTION_CACHE_FILE)
    except Exception as exc:
        logger.warning("Failed to persist portrait resolution cache: %s", exc)


def load_portrait_resolution_cache() -> None:
    if not PORTRAIT_RESOLUTION_CACHE_FILE.exists():
        return
    try:
        payload = json.loads(PORTRAIT_RESOLUTION_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load portrait resolution cache: %s", exc)
        return

    if isinstance(payload, dict):
        records = payload.get("entries", [])
    else:
        records = payload
    if not isinstance(records, list):
        return

    loaded = 0
    with portrait_resolution_lock:
        for record in records[-PORTRAIT_RESOLUTION_CACHE_MAX_ENTRIES:]:
            if not isinstance(record, dict):
                continue
            person_name_key = str(record.get("person_name_key", "")).strip().casefold()
            portrait_query_key = str(record.get("portrait_query_key", "")).strip().casefold()
            profession_hint = str(record.get("profession_hint", "")).strip().casefold()
            image_source = str(record.get("image_source", "")).strip()
            if not person_name_key or not portrait_query_key or not image_source:
                continue
            if not image_source.lower().startswith(("http://", "https://")):
                continue
            portrait_resolution_cache[
                (person_name_key, portrait_query_key, profession_hint)
            ] = (True, image_source)
            loaded += 1
    if loaded:
        logger.info(
            "Loaded %s portrait URL cache entries from %s",
            loaded,
            PORTRAIT_RESOLUTION_CACHE_FILE,
        )


def remember_successful_portrait_resolution(
    cache_key: tuple[str, str, str],
    image_source: str,
) -> None:
    with portrait_resolution_lock:
        portrait_resolution_cache.pop(cache_key, None)
        portrait_resolution_cache[cache_key] = (True, image_source)
    persist_portrait_resolution_cache()


def forget_portrait_resolution(cache_key: tuple[str, str, str]) -> None:
    removed = False
    with portrait_resolution_lock:
        removed = portrait_resolution_cache.pop(cache_key, None) is not None
    if removed:
        persist_portrait_resolution_cache()


load_portrait_resolution_cache()

SYSTEM_PROMPT = """
Generate exactly one round for an "Alive or Dead" guessing game.

Return only raw JSON with:
- `person_name`
- `actual_status`
- `date_of_birth`
- `date_of_death`
- `portrait_search_query`
- `next_round_label`
- `guessing_ui_html`
- `reveal_ui_html`

Round requirements:
- The prompt header will supply the exact locked person and, usually, the exact locked status to use.
- Do not pick a different person.
- Do not reinterpret, alias, shorten, or rewrite the supplied `person_name`.
- If a locked `actual_status` is supplied in the prompt header, use it exactly as given.
- Always return `date_of_birth` in `YYYY-MM-DD` format.
- If `actual_status` is `dead`, return `date_of_death` in `YYYY-MM-DD` format.
- If `actual_status` is `alive`, return `date_of_death` as `null`.
- Use real biographical dates that match the returned status. Do not guess inconsistent dates.
- Theme the UI around what the person is most famous for, but do not default to the same composition or design language every time.
- Make each round feel visually distinct. Vary composition, information density, panel structure, accents, pacing, and typography treatment from round to round.
- Choose the layout that best fits the person. It can be poster-like, editorial, tabloid, scoreboard-like, magazine-like, dossier-like, ticket-like, stage-like, or another strong visual direction.
- The main round card must stay comfortably wide on desktop and mobile. Do not collapse the entire guessing page into a skinny centered column.
- Push the creative treatment further than a normal app card. Be bold, thematic, and visually specific.
- The guessing page must include:
  - the person's name as the biggest text
  - one short status-neutral description of what they are most famous for
  - one short status-neutral fun fact
  - exactly one portrait `<img>` whose `src` is the exact literal `__PORTRAIT_IMAGE_URL__`
  - `submitGuess('alive')` and `submitGuess('dead')` buttons
- The reveal page must match the same theme and include `loadNextRound()`.
- The reveal page should include a compact factual date reference using the returned date fields, especially for the verdict moment, and should display those dates in a human-readable format like `April 5th, 2006` rather than raw ISO strings.
- Return `next_round_label` as a short, creative, celebrity-specific label for the `loadNextRound()` button. It should feel tailored to the subject's world, such as a catchphrase, encore, sequel cue, or related phrase, and should stay concise.
- The reveal page button text should use `next_round_label` exactly, not a generic `Next Round` label.

Output rules:
- Tailwind classes only. No <script>, <style>, <html>, <body>, <head>, markdown fences, SVG data URIs, or base64.
- Do not output any final remote image URL anywhere in the JSON. The backend will resolve the portrait URL from `portrait_search_query`.
- `portrait_search_query` must be plain text, not a URL, and should be a concise Wikimedia Commons portrait search phrase.
- Use a query structure like `[person name] [era or year if helpful] portrait`, `[person name] headshot`, or `[person name] publicity photo`.
- Prefer portrait, headshot, publicity photo, or press photo wording when useful.
- Do not include site names like `wikipedia`, `wikimedia`, or `commons` in `portrait_search_query`; the backend already searches Wikimedia Commons.
- If you are not confident you can suggest a clean portrait query, choose a different person instead of guessing.
- The guessing page must contain exactly one `<img>` whose `src` is the exact literal `__PORTRAIT_IMAGE_URL__`.
- The reveal page may omit the portrait, but if it includes one, it must reuse the exact same `__PORTRAIT_IMAGE_URL__` placeholder.
- Do not place `__PORTRAIT_IMAGE_URL__` anywhere except an `<img src>` attribute.
- The single portrait image may be used boldly, but on the guessing page it should still read as a recognizable portrait rather than disappearing into pure texture.
- Keep the actual portrait photo sharp and readable. Do not blur, heavily soften, smear, or defocus the `<img>` itself.
- Atmospheric treatments are allowed around the portrait with normal background layers, glows, gradients, or decorative shapes, but the actual portrait photo must remain crisp.
- Avoid turning the only guessing-page portrait into a totally abstract background wash unless the rest of the composition is especially strong and readable.
- Never repeat `__PORTRAIT_IMAGE_URL__` in CSS `url(...)`, inline styles, text, data attributes, masks, pseudo-background tricks, or additional `<img>` tags.
- Treat the portrait as one physical photo element. Build frames, glows, overlays, shadows, spotlights, and scenery around it with normal `<div>` layers, not by duplicating the portrait placeholder.
- Preserve the subject's face as a focal point. Keep the eyes and upper face clearly visible and readable in the final composition.
- Do not place large opaque text panels, badges, white fades, or heavy overlays across the eyes, forehead, or central face area.
- If text overlaps the portrait at all, place it in lower-third or side negative-space zones and keep the face unobstructed.
- Avoid crops or overlay treatments that make the portrait feel cut off at the eyes or washed out through the middle of the face.
- The full `person_name` must remain fully visible and readable. Never clip, crop, truncate, or push part of the name off-canvas.
- If the name is long, reduce the type size or wrap it cleanly instead of letting the text overflow outside its container.
- The name block must fit fully inside its panel with comfortable side padding. Do not let oversized display text run into the panel edge or off the right side of the card.
- For long names in split or poster layouts, prefer smaller headline sizes, more line breaks, or a wider text column instead of extreme oversized typography.
- Avoid giant serif or ultra-wide headline treatments for long names unless you are certain the full name still fits inside the visible card bounds on both desktop and mobile.
- Put the `submitGuess('alive')` and `submitGuess('dead')` buttons in a dedicated controls block outside the portrait area.
- The guess buttons must never overlap the portrait, float over the portrait, or appear inside the portrait panel.
- Do not use absolute or fixed positioning, inset utilities, negative margins, or translate utilities to place the guess buttons over the image.
- Do not use Tailwind aspect-ratio plugin classes like `aspect-w-*` or `aspect-h-*`; those utilities are not available here.
- If you need a square or shaped portrait area, use core Tailwind utilities like `aspect-square`, explicit `h-*`, `w-*`, `min-h-*`, and normal layout sizing instead.
- Do not use any remote URL or CSS `url(...)` inside backgrounds, overlays, masks, or decorative styles.
- Do not include extra `<img>` tags for logos, title cards, decorative textures, or background art.
- Be creatively ambitious within the HTML/Tailwind-only constraint. Use hierarchy, asymmetry, sections, badges, dividers, callouts, kicker labels, stat blocks, atmospheric color, and other layout devices when they fit.
- Built-in Tailwind animation and transition classes are allowed. Use a few purposeful motions like pulse, bounce, spin, shimmer-like movement, drifting lights, or reveal energy when they fit the theme.
- You may build thematic scenery and props from plain HTML/CSS structure alone: for example a TV frame for a TV star, stage lights or curtains for a performer, guitar-like silhouettes for a musician, marquees, tickets, tabloid bursts, scoreboards, dossiers, or dramatic spotlights.
- You may use absolute-positioned decorative layers, borders, glows, gradients, blur, rotations, and atmospheric shapes for non-button elements.
- Avoid repeating the same safe layout shell. Do not default to a generic centered card or the same two-column composition every round unless it is clearly the strongest fit for that person.
- Avoid repeating the same typography formula, same badge placement, same image treatment, or same information order every round.
- The reveal page should feel like a deliberate continuation or escalation of the guessing page concept, not just the same shell with new text.
- Keep each HTML fragment concise, but you may use richer structure and more layered presentation. Target under 3000 characters.
- Keep button text high contrast.
- Do not include literal backslash escape sequences like `\\n`, `\\r`, or `\\t` in the final HTML fragments.
- Design for a single-screen experience. The full guessing page and reveal page should fit comfortably in a normal desktop browser viewport without requiring vertical scrolling.
- Prefer compact spacing, restrained portrait sizing, and tighter composition when needed so the entire round is visible at once.
- Do not build tall poster layouts that push the buttons or key copy below the fold.

Spoiler rules for the guessing page:
- Never mention death, assassination, murder, funeral, memorial, burial, "late", "still alive", or any status-revealing phrasing.
- You may mention a movie, show, song, album, book, role, or franchise title even if the title contains a blocked word, but only if it is clearly the exact work title wrapped in quotes.
- If you mention a quoted work title with a blocked word, keep the surrounding description fully status-neutral and do not repeat the blocked word outside the quoted title.
- Example: for Angela Lansbury, `Most famous for playing Jessica Fletcher in "Murder, She Wrote"` is acceptable, but additional unquoted uses of `murder`, `death`, `dead`, `funeral`, `grave`, or similar spoiler wording are not.
- The guessing page description and fun fact must both be spoiler-safe after removing HTML tags and ignoring exact quoted work titles.
"""


CANDIDATE_SELECTION_PROMPT = """
Generate candidate people for one session of an "Alive or Dead" guessing game.

Return only raw JSON with:
- `candidates`

Candidate rules:
- Return exactly [CANDIDATE_COUNT] candidates.
- The backend will choose a 10-person round plan from your larger candidate list, so give a deep, varied bench instead of a short list of obvious staples.
- Each candidate must include:
  - `person_name`
  - `actual_status`
- `actual_status` must be exactly `alive` or `dead`. Never use `deceased`, `passed away`, `late`, or any other synonym.
- Pick famous people whose age would be between 8 and 120 years old whose status is genuinely guessable.
- Never return anyone from the forbidden list in the user prompt.
- Use canonical common public names only.
- Do not include duplicate people, aliases of the same person, or close variants of the same name.
- Prefer a diverse set of candidates instead of repeating the same obvious names.
- Do not default to the same obituary-canon or trivia-canon staples over and over.
- If many obvious celebrity staples are already forbidden, pivot hard into less overused but still broadly recognizable famous people.
- Spread the list across different eras and domains such as film, television, music, sports, public life, and culture rather than clustering around the same few classic dead celebrities.

Critical forbidden-list compliance:
- The forbidden list is authoritative and overrides every other instruction.
- Before you write JSON, compare every candidate against every forbidden entry case-insensitively.
- If a candidate is an exact match, near match, alias, nickname, title variation, stage name variation, spelling variation, or you are not fully sure they are different, discard them and choose a different person.
- Treat uncertainty as forbidden.
- Do not output a forbidden person.
"""


def extract_json(text: str) -> dict | list | None:
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


def trim_fifo_names(names: list[str], max_names: int) -> list[str]:
    if max_names <= 0:
        return []
    return list(names[-max_names:])


def normalize_client_ip(raw_ip: str | None) -> str:
    normalized = " ".join(str(raw_ip or "").split())
    return normalized.casefold() if normalized else "unknown"


def extract_client_ip(request: FastAPIRequest) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def read_ip_histories_unlocked() -> dict[str, list[str]]:
    try:
        if IP_HISTORY_FILE.exists():
            content = IP_HISTORY_FILE.read_text(encoding="utf-8").strip()
            if content:
                payload = json.loads(content)
                if isinstance(payload, dict):
                    histories: dict[str, list[str]] = {}
                    for raw_ip, raw_names in payload.items():
                        if not isinstance(raw_names, list):
                            continue
                        histories[normalize_client_ip(raw_ip)] = trim_fifo_names(
                            merge_names_recency_preserving([], raw_names),
                            IP_HISTORY_MAX_NAMES,
                        )
                    return histories
    except Exception as exc:
        logger.error("Failed to read IP history: %s", exc)
    return {}


def write_ip_histories_unlocked(histories: dict[str, list[str]]) -> None:
    normalized_histories: dict[str, list[str]] = {}
    for raw_ip, raw_names in histories.items():
        ip_key = normalize_client_ip(raw_ip)
        if not ip_key:
            continue
        normalized_histories[ip_key] = trim_fifo_names(
            merge_names_recency_preserving([], raw_names),
            IP_HISTORY_MAX_NAMES,
        )
    IP_HISTORY_FILE.write_text(
        json.dumps(normalized_histories, ensure_ascii=True),
        encoding="utf-8",
    )


def get_ip_history(client_ip_key: str) -> list[str]:
    with ip_history_lock:
        histories = read_ip_histories_unlocked()
        return list(histories.get(normalize_client_ip(client_ip_key), []))


def update_ip_history(client_ip_key: str, names: list[str]) -> list[str]:
    ip_key = normalize_client_ip(client_ip_key)
    with ip_history_lock:
        histories = read_ip_histories_unlocked()
        history = merge_names_recency_preserving(histories.get(ip_key, []), names)
        history = trim_fifo_names(history, IP_HISTORY_MAX_NAMES)
        histories[ip_key] = history
        write_ip_histories_unlocked(histories)
    for name in names:
        if name:
            forget_candidate_from_ip_pool(ip_key, name)
    return list(history)


def candidate_bank_entry_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def prune_candidate_like_entries(entries: list[dict], ttl_seconds: int, max_entries: int) -> list[dict]:
    if not entries:
        return []

    cutoff_ts = time.time() - max(ttl_seconds, 0)
    deduped: list[dict] = []
    seen: set[str] = set()

    for raw_entry in reversed(entries):
        if not isinstance(raw_entry, dict):
            continue
        person_name = " ".join(str(raw_entry.get("person_name", "")).split())
        if not person_name:
            continue
        actual_status = str(raw_entry.get("actual_status", "")).strip().lower()
        if actual_status not in {"alive", "dead"}:
            continue
        normalized_key = normalize_name_key(person_name)
        if normalized_key in seen:
            continue
        updated_at_raw = str(raw_entry.get("updated_at", "")).strip()
        updated_at_dt: datetime | None = None
        if updated_at_raw:
            try:
                updated_at_dt = datetime.fromisoformat(updated_at_raw.replace("Z", "+00:00"))
            except ValueError:
                updated_at_dt = None
        if updated_at_dt is not None and updated_at_dt.timestamp() < cutoff_ts:
            continue
        seen.add(normalized_key)
        deduped.append(
            {
                "person_name": person_name,
                "actual_status": actual_status,
                "updated_at": updated_at_dt.isoformat() if updated_at_dt else candidate_bank_entry_timestamp(),
            }
        )

    deduped.reverse()
    if len(deduped) > max_entries:
        deduped = deduped[-max_entries:]
    return deduped


def read_ip_candidate_pools_unlocked() -> dict[str, list[dict]]:
    try:
        if CANDIDATE_IP_POOL_FILE.exists():
            content = CANDIDATE_IP_POOL_FILE.read_text(encoding="utf-8").strip()
            if content:
                payload = json.loads(content)
                if isinstance(payload, dict):
                    normalized: dict[str, list[dict]] = {}
                    for raw_ip_key, raw_entries in payload.items():
                        if not isinstance(raw_entries, list):
                            continue
                        ip_key = normalize_client_ip(str(raw_ip_key))
                        normalized_entries = prune_candidate_like_entries(
                            raw_entries,
                            IP_CANDIDATE_POOL_TTL_SECONDS,
                            IP_CANDIDATE_POOL_MAX_ENTRIES,
                        )
                        if normalized_entries:
                            normalized[ip_key] = normalized_entries
                    return normalized
    except Exception as exc:
        logger.error("Failed to read IP candidate pools: %s", exc)
    return {}


def write_ip_candidate_pools_unlocked(pools: dict[str, list[dict]]) -> None:
    normalized: dict[str, list[dict]] = {}
    for raw_ip_key, raw_entries in pools.items():
        if not isinstance(raw_entries, list):
            continue
        ip_key = normalize_client_ip(str(raw_ip_key))
        normalized_entries = prune_candidate_like_entries(
            raw_entries,
            IP_CANDIDATE_POOL_TTL_SECONDS,
            IP_CANDIDATE_POOL_MAX_ENTRIES,
        )
        if normalized_entries:
            normalized[ip_key] = normalized_entries
    CANDIDATE_IP_POOL_FILE.write_text(
        json.dumps(normalized, ensure_ascii=True),
        encoding="utf-8",
    )


def select_candidates_from_ip_pool(client_ip_key: str, forbidden: list[str], limit: int) -> list[LockedCandidate]:
    if limit <= 0:
        return []
    ip_key = normalize_client_ip(client_ip_key)
    forbidden_keys = {normalize_name_key(name) for name in forbidden}
    selected: list[LockedCandidate] = []
    selected_keys: set[str] = set()
    with candidate_ip_pool_lock:
        pools = read_ip_candidate_pools_unlocked()
        entries = list(pools.get(ip_key, []))

    for entry in reversed(entries):
        person_name = str(entry.get("person_name", ""))
        actual_status = str(entry.get("actual_status", "")).strip().lower() or None
        key = normalize_name_key(person_name)
        if not person_name or key in forbidden_keys or key in selected_keys:
            continue
        selected_keys.add(key)
        selected.append(
            LockedCandidate(
                person_name=person_name,
                actual_status=actual_status,
                source="ip_candidate_pool",
            )
        )
        if len(selected) >= limit:
            break

    return selected


def remember_candidates_for_ip_pool(client_ip_key: str, candidates: list[LockedCandidate]) -> None:
    if not candidates:
        return
    ip_key = normalize_client_ip(client_ip_key)
    new_entries = [
        {
            "person_name": " ".join(candidate.person_name.split()),
            "actual_status": (candidate.actual_status or "").strip().lower(),
            "updated_at": candidate_bank_entry_timestamp(),
        }
        for candidate in candidates
        if candidate.person_name and (candidate.actual_status or "").strip().lower() in {"alive", "dead"}
    ]
    if not new_entries:
        return
    with candidate_ip_pool_lock:
        pools = read_ip_candidate_pools_unlocked()
        entries = list(pools.get(ip_key, []))
        entries.extend(new_entries)
        pools[ip_key] = entries
        write_ip_candidate_pools_unlocked(pools)


def forget_candidate_from_ip_pool(client_ip_key: str, person_name: str) -> None:
    ip_key = normalize_client_ip(client_ip_key)
    person_key = normalize_name_key(person_name)
    with candidate_ip_pool_lock:
        pools = read_ip_candidate_pools_unlocked()
        entries = pools.get(ip_key, [])
        filtered = [
            entry
            for entry in entries
            if normalize_name_key(str(entry.get("person_name", ""))) != person_key
        ]
        if filtered:
            pools[ip_key] = filtered
        elif ip_key in pools:
            pools.pop(ip_key, None)
        if len(filtered) != len(entries):
            write_ip_candidate_pools_unlocked(pools)


def prune_candidate_bank_entries(entries: list[dict]) -> list[dict]:
    return prune_candidate_like_entries(
        entries,
        CANDIDATE_BANK_TTL_SECONDS,
        CANDIDATE_BANK_MAX_ENTRIES,
    )


def read_candidate_bank_unlocked() -> list[dict]:
    try:
        if CANDIDATE_BANK_FILE.exists():
            content = CANDIDATE_BANK_FILE.read_text(encoding="utf-8").strip()
            if content:
                payload = json.loads(content)
                if isinstance(payload, list):
                    return prune_candidate_bank_entries(payload)
    except Exception as exc:
        logger.error("Failed to read candidate bank: %s", exc)
    return []


def write_candidate_bank_unlocked(entries: list[dict]) -> None:
    normalized_entries = prune_candidate_bank_entries(entries)
    CANDIDATE_BANK_FILE.write_text(
        json.dumps(normalized_entries, ensure_ascii=True),
        encoding="utf-8",
    )


def select_candidates_from_bank(forbidden: list[str], limit: int) -> list[LockedCandidate]:
    if limit <= 0:
        return []
    forbidden_keys = {normalize_name_key(name) for name in forbidden}
    selected: list[LockedCandidate] = []
    selected_keys: set[str] = set()
    with candidate_bank_lock:
        entries = read_candidate_bank_unlocked()

    for entry in reversed(entries):
        person_name = str(entry.get("person_name", ""))
        actual_status = str(entry.get("actual_status", "")).strip().lower() or None
        key = normalize_name_key(person_name)
        if not person_name or key in forbidden_keys or key in selected_keys:
            continue
        selected_keys.add(key)
        selected.append(
            LockedCandidate(
                person_name=person_name,
                actual_status=actual_status,
                source="candidate_bank",
            )
        )
        if len(selected) >= limit:
            break

    return selected


def remember_candidates_in_bank(candidates: list[LockedCandidate]) -> None:
    if not candidates:
        return
    new_entries = [
        {
            "person_name": " ".join(candidate.person_name.split()),
            "actual_status": (candidate.actual_status or "").strip().lower(),
            "updated_at": candidate_bank_entry_timestamp(),
        }
        for candidate in candidates
        if candidate.person_name and (candidate.actual_status or "").strip().lower() in {"alive", "dead"}
    ]
    if not new_entries:
        return
    with candidate_bank_lock:
        entries = read_candidate_bank_unlocked()
        entries.extend(new_entries)
        write_candidate_bank_unlocked(entries)


def forget_candidate_from_bank(person_name: str) -> None:
    person_key = normalize_name_key(person_name)
    with candidate_bank_lock:
        entries = read_candidate_bank_unlocked()
        filtered = [
            entry
            for entry in entries
            if normalize_name_key(str(entry.get("person_name", ""))) != person_key
        ]
        if len(filtered) != len(entries):
            write_candidate_bank_unlocked(filtered)


def url_targets_wikimedia(url: str) -> bool:
    lowered = url.lower()
    return (
        "wikimedia.org" in lowered
        or "wikipedia.org" in lowered
        or "wikidata.org" in lowered
    )


def wait_for_wikimedia_request_slot(url: str) -> None:
    if not url_targets_wikimedia(url):
        return

    global last_wikimedia_request_at
    min_interval_seconds = max(WIKIMEDIA_MIN_REQUEST_INTERVAL_MS, 0.0) / 1000.0
    if min_interval_seconds <= 0:
        return

    with wikimedia_request_lock:
        elapsed = time.monotonic() - last_wikimedia_request_at
        remaining = min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        last_wikimedia_request_at = time.monotonic()


def collect_ip_reserved_names(client_ip_key: str, exclude_session_id: str | None = None) -> list[str]:
    reserved_names: list[str] = []
    normalized_ip = normalize_client_ip(client_ip_key)
    for session_id, session in sessions.items():
        if exclude_session_id and session_id == exclude_session_id:
            continue
        try:
            round_number = int(session.get("round_number", 0))
        except (TypeError, ValueError):
            continue
        if round_number >= ROUNDS_PER_GAME:
            continue
        if normalize_client_ip(session.get("client_ip_key")) != normalized_ip:
            continue
        reserved_names.extend(session.get("reserved_names", []))
    return merge_names_recency_preserving([], reserved_names)


def build_generation_prompt(
    retry_notes: list[str] | None = None,
    locked_person_name: str | None = None,
    locked_actual_status: str | None = None,
) -> str:
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

    prompt_body = locked_section + SYSTEM_PROMPT
    return f"{prompt_body}\n\n{retry_section}Return one complete round only."


def build_candidate_selection_prompt(
    forbidden: list[str],
    candidate_count: int,
    retry_notes: list[str] | None = None,
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

    prompt_body = CANDIDATE_SELECTION_PROMPT.replace("[CANDIDATE_COUNT]", str(candidate_count))
    return (
        "CRITICAL FORBIDDEN LIST:\n"
        f"[{forbidden_tail}]\n\n"
        f"{prompt_body}\n\n"
        f"{retry_section}"
        "Return one candidate-selection JSON object only."
    )


def build_retry_note(error_message: str) -> str:
    lowered = error_message.lower()
    if "actual_status" in lowered and "deceased" in lowered:
        return (
            "Your previous attempt used the invalid status word 'deceased'. "
            "Use exactly `alive` or `dead` for `actual_status`; never use synonyms."
        )
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
    if (
        "Candidate selection returned no allowed person" in error_message
        or "Candidate negotiation returned no newly allowed people" in error_message
    ):
        rejected_tail = ""
        if "Rejected candidates:" in error_message:
            rejected_tail = error_message.split("Rejected candidates:", 1)[1].strip()
        return (
            "Your previous candidate list did not contain any allowed person. "
            "Return a much more diverse candidate list, avoid obvious or previously used names, and go deeper than the usual celebrity staples. "
            + (
                f"Do not reuse these rejected examples: {rejected_tail}. "
                if rejected_tail and rejected_tail.lower() != "none"
                else ""
            )
        )
    if "date_of_birth" in error_message or "date_of_death" in error_message:
        return (
            "Your previous attempt returned inconsistent or invalid date fields. "
            "Return `date_of_birth` in YYYY-MM-DD format, return `date_of_death` only for dead people, use `null` for alive people, and keep the dates consistent with `actual_status`."
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
            f"Use exactly one portrait `<img src>` with the exact literal `{PORTRAIT_IMAGE_PLACEHOLDER}`. "
            "Do not place the placeholder in CSS `url(...)`, inline styles, text, data attributes, or extra `<img>` tags. "
            "If you want a background-like portrait treatment, style that one `<img>` with layout classes and surrounding decorative `<div>` layers instead of repeating the placeholder. "
            "Do not output a final remote image URL yourself, and if the reveal page also shows a portrait, reuse the same single-placeholder pattern there."
        )
    if "portrait presentation" in error_message.lower() or "guessing portrait" in error_message.lower():
        return (
            "Your previous attempt made the guessing portrait too abstract or unreadable. "
            "Keep the single guessing-page portrait sharp, recognizable, and comfortably framed. "
            "Do not blur it, do not fade it heavily with low opacity, and do not zoom so aggressively that the head is truncated."
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


def trim_retry_notes(retry_notes: list[str], limit: int = 4) -> list[str]:
    deduped = list(dict.fromkeys(note.strip() for note in retry_notes if note.strip()))
    if limit <= 0:
        return []
    return deduped[-limit:]


def extract_visible_text(fragment: str) -> str:
    text = HTML_TAG_RE.sub(" ", fragment)
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_generated_html_fragment(fragment: str) -> str:
    normalized = str(fragment or "")
    normalized = normalized.replace("\\r", "\n").replace("\\n", "\n").replace("\\t", " ")
    return normalized.strip()


def wrap_round_fragment(fragment: str) -> str:
    normalized = normalize_generated_html_fragment(fragment)
    if not normalized:
        return normalized
    return f"<div class='round-fragment w-full'>{normalized}</div>"


def format_display_date(iso_date: str) -> str:
    parsed = datetime.strptime(iso_date, "%Y-%m-%d")
    day = parsed.day
    suffix = "th"
    if day % 100 not in {11, 12, 13}:
        if day % 10 == 1:
            suffix = "st"
        elif day % 10 == 2:
            suffix = "nd"
        elif day % 10 == 3:
            suffix = "rd"
    return f"{parsed.strftime('%B')} {day}{suffix}, {parsed.year}"


def humanize_reveal_dates(fragment: str, date_of_birth: str, date_of_death: str | None) -> str:
    rewritten = fragment
    replacements = [date_of_birth]
    if date_of_death:
        replacements.append(date_of_death)
    for iso_date in replacements:
        rewritten = rewritten.replace(iso_date, format_display_date(iso_date))
    return rewritten


NEXT_ROUND_BUTTON_RE = re.compile(
    r"<(?P<tag>button|a)\b(?P<attrs>[^>]*\bonclick\s*=\s*(?P<q>['\"])loadNextRound\(\)(?P=q)[^>]*)>(?P<label>.*?)</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)


def sanitize_next_round_label(label: str | None) -> str:
    normalized = " ".join(str(label or "").split())
    normalized = html.escape(normalized or "Next Round", quote=False)
    if len(normalized) > 80:
        normalized = normalized[:80].rstrip()
    return normalized or "Next Round"


def add_classes_to_attrs(attrs: str, extra_classes: str) -> str:
    class_match = re.search(r"\bclass\s*=\s*(['\"])(.*?)\1", attrs, re.IGNORECASE | re.DOTALL)
    if class_match:
        quote = class_match.group(1)
        existing_classes = class_match.group(2).strip()
        combined_classes = " ".join(
            part for part in (existing_classes, extra_classes.strip()) if part
        )
        replacement = f'class={quote}{combined_classes}{quote}'
        return attrs[: class_match.start()] + replacement + attrs[class_match.end() :]
    return f"{attrs} class=\"{extra_classes.strip()}\""


def rewrite_next_round_control(fragment: str, label: str, hide_on_mobile: bool = False) -> str:
    button_label = sanitize_next_round_label(label)

    def repl(match: re.Match[str]) -> str:
        tag = match.group("tag")
        attrs = match.group("attrs")
        if hide_on_mobile:
            attrs = add_classes_to_attrs(attrs, "hidden md:inline-flex")
        return f"<{tag}{attrs}>{button_label}</{tag}>"

    return NEXT_ROUND_BUTTON_RE.sub(repl, fragment, count=1)


def build_mobile_next_round_bar(label: str) -> str:
    button_label = sanitize_next_round_label(label)
    return (
        "<div class='fixed inset-x-0 bottom-0 z-50 flex justify-center px-4 pb-4 pt-3 md:hidden' "
        "style='padding-bottom: calc(1rem + env(safe-area-inset-bottom));'>"
        "<div class='pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-black via-black/80 to-transparent'></div>"
        "<button onclick=\"loadNextRound()\" "
        "class='pointer-events-auto relative rounded-full border border-white/10 bg-white px-6 py-4 text-sm font-black uppercase tracking-[0.2em] text-black shadow-2xl shadow-black/40 transition hover:scale-105 active:scale-95'>"
        f"{button_label}"
        "</button>"
        "</div>"
    )


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


def extract_context_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize_identity_text(text)
        if len(token) > 2
        and token not in NAME_TOKEN_STOPWORDS
        and token not in CONTEXT_STOPWORDS
        and token not in GENERIC_PORTRAIT_QUERY_TOKENS
    }


def score_context_families(context_text: str) -> dict[str, int]:
    context_tokens = extract_context_tokens(context_text)
    return {
        family: sum(1 for token in tokens if token in context_tokens)
        for family, tokens in CONTEXT_FAMILY_KEYWORDS.items()
    }


def infer_context_family(context_text: str) -> str | None:
    family_scores = score_context_families(context_text)
    if not family_scores:
        return None
    family, score = max(family_scores.items(), key=lambda item: item[1])
    return family if score > 0 else None


def infer_profession_hint(context_text: str) -> str | None:
    family = infer_context_family(context_text)
    if family == "acting":
        return "actor"
    if family == "music":
        return "musician"
    if family == "sports":
        return "athlete"
    if family == "politics":
        return "politician"
    if family == "science":
        return "scientist"
    return None


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


def page_looks_non_person(title: str, description: str = "") -> bool:
    title_tokens = set(tokenize_identity_text(title))
    description_tokens = set(tokenize_identity_text(description))
    tokens = title_tokens | description_tokens
    if tokens & NON_PERSON_PAGE_KEYWORDS:
        return True
    if "ocean liner" in description.casefold() or "cruise ship" in description.casefold():
        return True
    return bool(
        {"ship", "liner", "vessel", "ferry", "boat", "yacht", "submarine", "aircraft", "plane", "train", "vehicle"}
        & tokens
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


def score_wikipedia_page_result(page: dict, person_name: str, portrait_context: str) -> int:
    title = str(page.get("title", ""))
    description = str(page.get("description", ""))
    combined_tokens = set(tokenize_identity_text(title)) | set(tokenize_identity_text(description))
    normalized_person_name = normalize_identity_text(person_name)
    matches_person, matched_count, _, _ = analyze_commons_title_match(title, person_name)
    context_tokens = extract_context_tokens(portrait_context)
    context_family = infer_context_family(portrait_context)

    score = 0
    if page_looks_non_person(title, description):
        score -= 120
    if title_looks_non_biographical(title):
        score -= 100
    if matches_person:
        score += 30
    score += matched_count * 5
    normalized_title = normalize_identity_text(title)
    if normalized_title == normalized_person_name:
        score += 8
    if normalized_title.startswith(normalized_person_name + " ("):
        score += 12

    overlap = len(context_tokens & combined_tokens)
    score += min(overlap, 6) * 4

    if context_family:
        family_tokens = CONTEXT_FAMILY_KEYWORDS.get(context_family, set())
        score += len(family_tokens & combined_tokens) * 5
        mismatch_tokens = PAGE_ROLE_MISMATCH_KEYWORDS.get(context_family, set())
        if mismatch_tokens:
            score -= len(mismatch_tokens & combined_tokens) * 8

    return score


def normalize_name_key(name: str) -> str:
    return " ".join(str(name).split()).casefold()


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


def iter_node_ancestors(node: FragmentNode):
    current: FragmentNode | None = node
    while current is not None:
        yield current
        current = current.parent


def tailwind_opacity_value(token: str) -> float | None:
    match = TAILWIND_OPACITY_RE.match(token)
    if not match:
        return None
    return int(match.group(1)) / 100.0


def tailwind_scale_factor(token: str) -> float | None:
    match = TAILWIND_SCALE_RE.match(token)
    if not match:
        return None
    return int(match.group(1)) / 100.0


def node_chain_uses_blur(node: FragmentNode) -> bool:
    for current in iter_node_ancestors(node):
        for token in split_tailwind_classes(current):
            if token == "blur" or token.startswith("blur-"):
                return True
        style = current.attrs.get("style", "")
        if INLINE_BLUR_RE.search(style):
            return True
    return False


def node_chain_lowest_opacity(node: FragmentNode) -> float:
    lowest = 1.0
    for current in iter_node_ancestors(node):
        for token in split_tailwind_classes(current):
            opacity = tailwind_opacity_value(token)
            if opacity is not None:
                lowest = min(lowest, opacity)
        style = current.attrs.get("style", "")
        for match in INLINE_OPACITY_RE.finditer(style):
            try:
                lowest = min(lowest, float(match.group(1)))
            except ValueError:
                continue
    return lowest


def node_chain_largest_scale(node: FragmentNode) -> float:
    largest = 1.0
    for current in iter_node_ancestors(node):
        for token in split_tailwind_classes(current):
            scale = tailwind_scale_factor(token)
            if scale is not None:
                largest = max(largest, scale)
        style = current.attrs.get("style", "")
        match = INLINE_SCALE_RE.search(style)
        if match:
            try:
                largest = max(largest, float(match.group(1)))
            except ValueError:
                continue
    return largest


def validate_guessing_portrait_presentation(fragment: str, person_name: str) -> None:
    root = parse_fragment_tree(fragment)
    portrait_nodes = [
        node
        for node in iter_fragment_nodes(root)
        if node.tag == "img"
        and node.attrs.get("src", "").strip() == PORTRAIT_IMAGE_PLACEHOLDER
    ]

    if len(portrait_nodes) != 1:
        return

    portrait_node = portrait_nodes[0]
    if node_chain_uses_blur(portrait_node):
        raise ValueError(
            f"Guessing portrait presentation is too blurred for {person_name}"
        )

    lowest_opacity = node_chain_lowest_opacity(portrait_node)
    if lowest_opacity < GUESSING_PORTRAIT_MIN_OPACITY:
        raise ValueError(
            f"Guessing portrait presentation is too faint for {person_name}"
        )

    largest_scale = node_chain_largest_scale(portrait_node)
    if largest_scale > GUESSING_PORTRAIT_MAX_SCALE:
        raise ValueError(
            f"Guessing portrait presentation is too tightly cropped or zoomed for {person_name}"
        )


def extract_wikimedia_thumbnail_width(image_source: str) -> str | None:
    match = re.search(r"/(\d+)px-[^/?#]+(?:[?#].*)?$", image_source)
    if match:
        return match.group(1)
    return None


def fetch_json_url(url: str, timeout_seconds: float, user_agent: str) -> dict:
    request = UrlRequest(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        wait_for_wikimedia_request_slot(url)
        with urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", None) or response.getcode()
            if status and status >= 400:
                raise RuntimeError(f"HTTP {status}")
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP Error {exc.code}: {exc.reason}") from exc
    return json.loads(payload)


def is_rate_limited_error(error: BaseException | str) -> bool:
    message = str(error)
    return "429" in message or "Too Many Requests" in message


def portrait_retry_backoff_seconds(attempt_index: int) -> float:
    base_seconds = max(WIKIMEDIA_429_RETRY_BACKOFF_MS, 0.0) / 1000.0
    return base_seconds * (2 ** attempt_index)


def validate_image_url_reachable(image_source: str, person_name: str, label: str) -> None:
    if not IMAGE_URL_VALIDATION_ENABLED:
        return

    with image_url_validation_lock:
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
            wait_for_wikimedia_request_slot(image_source)
            with urlopen(request, timeout=IMAGE_URL_VALIDATION_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", None) or response.getcode()
                content_type = str(response.headers.get_content_type()).lower()
                if status and status >= 400:
                    failure_reason = f"{method} returned HTTP {status}"
                    continue
                if not content_type.startswith("image/"):
                    failure_reason = f"{method} returned content type {content_type or 'unknown'}"
                    continue
                with image_url_validation_lock:
                    image_url_validation_cache[image_source] = (True, "ok")
                return
        except Exception as exc:
            failure_reason = f"{method} failed: {exc}"

    with image_url_validation_lock:
        image_url_validation_cache[image_source] = (False, failure_reason)
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


def build_portrait_search_candidates(
    person_name: str,
    portrait_search_query: str,
    portrait_context: str = "",
) -> list[str]:
    profession_hint = infer_profession_hint(portrait_context)
    candidates = []
    if profession_hint:
        candidates.extend(
            [
                f"{person_name} {profession_hint} portrait",
                f"{person_name} {profession_hint} publicity photo",
            ]
        )
    candidates.extend(
        [
            portrait_search_query,
            f"{person_name} portrait",
            f"{person_name} headshot",
            f"{person_name} publicity portrait",
        ]
    )
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


def get_wikipedia_page_metadata(page_title: str) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "redirects": "1",
        "titles": page_title,
        "prop": "pageprops|categories",
        "cllimit": "max",
    }
    data = fetch_json_url(
        f"{WIKIPEDIA_ACTION_API_URL}?{urlencode(params)}",
        timeout_seconds=WIKIMEDIA_SEARCH_TIMEOUT_SECONDS,
        user_agent=WIKIMEDIA_API_USER_AGENT,
    )
    pages = data.get("query", {}).get("pages", {})
    if isinstance(pages, dict):
        for page in pages.values():
            if isinstance(page, dict):
                return page
    return {}


def get_wikidata_entity(entity_id: str) -> dict:
    encoded_id = quote(entity_id, safe="")
    return fetch_json_url(
        f"{WIKIDATA_ENTITY_DATA_URL}{encoded_id}.json",
        timeout_seconds=WIKIMEDIA_SEARCH_TIMEOUT_SECONDS,
        user_agent=WIKIMEDIA_API_USER_AGENT,
    )


def infer_status_from_wikipedia_categories(page_data: dict) -> str | None:
    categories = page_data.get("categories") or []
    category_titles = [
        str(category.get("title", ""))
        for category in categories
        if isinstance(category, dict)
    ]
    lowered_titles = [title.casefold() for title in category_titles]
    if any(title.endswith("living people") for title in lowered_titles):
        return "alive"
    if any(re.search(r"category:\d{4} deaths$", title) for title in lowered_titles):
        return "dead"
    return None


def infer_status_from_wikidata_entity(entity_data: dict) -> str | None:
    entities = entity_data.get("entities", {})
    if not isinstance(entities, dict):
        return None
    for entity in entities.values():
        if not isinstance(entity, dict):
            continue
        claims = entity.get("claims", {})
        if not isinstance(claims, dict):
            continue
        if claims.get("P570"):
            return "dead"
    return None


def infer_status_from_summary(summary: dict) -> str | None:
    extract = " ".join(str(summary.get("extract", "")).split())
    if not extract:
        return None
    prefix = extract[:240].casefold()
    if " was " in prefix and "(born" not in prefix:
        return "dead"
    if " is " in prefix:
        return "alive"
    return None


def resolve_person_actual_status(person_name: str) -> str | None:
    cache_key = normalize_name_key(person_name)
    with status_resolution_lock:
        if cache_key in status_resolution_cache:
            return status_resolution_cache[cache_key]

    resolved_status: str | None = None
    try:
        ranked_pages = sorted(
            search_wikipedia_pages(person_name),
            key=lambda page: (
                -score_wikipedia_page_result(page, person_name, ""),
                -len(str(page.get("description", ""))),
            ),
        )
        for page in ranked_pages:
            title = str(page.get("title", "")).strip()
            description = str(page.get("description", "")).strip()
            if not title or title_looks_non_biographical(title):
                continue
            if page_looks_non_person(title, description):
                continue
            matches_person, _, _, _ = analyze_commons_title_match(title, person_name)
            if not matches_person and not normalize_identity_text(title).startswith(
                normalize_identity_text(person_name)
            ):
                continue

            page_data = get_wikipedia_page_metadata(title)
            resolved_status = infer_status_from_wikipedia_categories(page_data)
            if resolved_status:
                break

            pageprops = page_data.get("pageprops") or {}
            wikibase_item = str(pageprops.get("wikibase_item", "")).strip()
            if wikibase_item:
                try:
                    entity_data = get_wikidata_entity(wikibase_item)
                except Exception:
                    entity_data = {}
                resolved_status = infer_status_from_wikidata_entity(entity_data)
                if resolved_status:
                    break

            page_key = str(page.get("key", "")).strip()
            if page_key:
                try:
                    summary = get_wikipedia_page_summary(page_key)
                except Exception:
                    summary = {}
                resolved_status = infer_status_from_summary(summary)
                if resolved_status:
                    break
    except Exception as exc:
        logger.warning("Status verification failed for %s: %s", person_name, exc)

    with status_resolution_lock:
        status_resolution_cache[cache_key] = resolved_status
    return resolved_status


def resolve_wikipedia_page_portrait_url(
    person_name: str,
    portrait_search_query: str,
    portrait_context: str = "",
) -> tuple[str, str] | None:
    profession_hint = infer_profession_hint(portrait_context)
    search_terms = [portrait_search_query]
    if profession_hint:
        search_terms.append(f"{person_name} {profession_hint}")
    search_terms.append(person_name)

    pages_by_key: dict[str, dict] = {}
    for search_term in search_terms:
        for page in search_wikipedia_pages(search_term):
            key = str(page.get("key", "")).strip() or str(page.get("title", "")).strip()
            if not key:
                continue
            pages_by_key.setdefault(key, page)

    ranked_pages = sorted(
        pages_by_key.values(),
        key=lambda page: (
            -score_wikipedia_page_result(page, person_name, portrait_context),
            -len(str(page.get("description", ""))),
        ),
    )

    for page in ranked_pages:
        title = str(page.get("title", ""))
        description = str(page.get("description", ""))
        if title_looks_non_biographical(title):
            continue
        if page_looks_non_person(title, description):
            continue
        matches_person, _, _, _ = analyze_commons_title_match(title, person_name)
        if not matches_person and not normalize_identity_text(title).startswith(
            normalize_identity_text(person_name)
        ):
            continue
        page_key = str(page.get("key", "")).strip()
        if not page_key:
            continue
        try:
            summary = get_wikipedia_page_summary(page_key)
        except Exception:
            continue
        summary_title = str(summary.get("title", ""))
        summary_description = str(
            summary.get("description")
            or summary.get("extract")
            or ""
        )
        if page_looks_non_person(summary_title or title, summary_description):
            continue
        thumbnail = summary.get("thumbnail") or {}
        image_source = strip_url_query_and_fragment(str(thumbnail.get("source", "")).strip())
        if not image_source:
            continue
        lowered = image_source.casefold()
        if lowered.startswith("//"):
            image_source = f"https:{image_source}"
            lowered = image_source.casefold()
        if not lowered.startswith(REMOTE_IMAGE_PREFIXES):
            continue
        if any(marker in lowered for marker in WIKIMEDIA_DOMAIN_MARKERS):
            if not any(lowered.startswith(prefix) for prefix in WIKIMEDIA_THUMB_PREFIXES):
                continue
            width = extract_wikimedia_thumbnail_width(image_source)
            if width not in VALID_WIKIMEDIA_THUMB_WIDTHS:
                continue
        if not url_has_supported_raster_extension(image_source):
            continue
        return image_source, title

    return None


def resolve_wikimedia_portrait_url(
    person_name: str,
    portrait_search_query: str,
    portrait_context: str = "",
) -> str:
    profession_hint = (infer_profession_hint(portrait_context) or "").casefold()
    cache_key = (person_name.casefold(), portrait_search_query.casefold(), profession_hint)
    with portrait_resolution_lock:
        cached = portrait_resolution_cache.get(cache_key)
    if cached is not None:
        ok, value = cached
        cache_invalidated = False
        if ok:
            if is_local_portrait_fallback_url(str(value)):
                logger.info(
                    "Evicting cached local portrait fallback for %s so real portrait resolution can retry",
                    person_name,
                )
                forget_portrait_resolution(cache_key)
                cache_invalidated = True
            elif str(value).lower().startswith(("http://", "https://")):
                try:
                    validate_image_url_reachable(value, person_name, "Cached portrait")
                except Exception as exc:
                    logger.warning(
                        "Evicting stale cached portrait URL for %s: %s",
                        person_name,
                        exc,
                    )
                    forget_portrait_resolution(cache_key)
                    cache_invalidated = True
                else:
                    return value
            else:
                forget_portrait_resolution(cache_key)
                cache_invalidated = True
        if not cache_invalidated:
            raise ValueError(value)

    search_query = validate_portrait_search_query(person_name, portrait_search_query)
    started_at = perf_now()
    max_attempts = 1 if PORTRAIT_FALLBACK_MODE == "fast" else max(
        WIKIMEDIA_429_RETRY_COUNT + 1,
        1,
    )
    last_rate_limit_error: str | None = None
    attempted_queries_for_log: list[str] = []
    resolution_error = (
        f"No valid Wikimedia portrait found for {person_name} using portrait_search_query '{search_query}'"
    )

    for rate_limit_attempt in range(max_attempts):
        attempted_queries: list[str] = []
        tried_urls: set[str] = set()
        try:
            wikipedia_page_result = resolve_wikipedia_page_portrait_url(
                person_name,
                search_query,
                portrait_context,
            )
            if wikipedia_page_result is not None:
                image_source, resolved_title = wikipedia_page_result
                validate_image_url_reachable(image_source, person_name, "Resolved portrait")
                remember_successful_portrait_resolution(cache_key, image_source)
                append_gemini_audit_log(
                    "portrait.resolve",
                    status="accepted",
                    person_name=person_name,
                    portrait_search_query=search_query,
                    attempted_queries=["wikipedia_page_exact_or_search"],
                    resolved_portrait_url=image_source,
                    resolved_from_query="wikipedia_page_exact_or_search",
                    resolved_title=resolved_title,
                )
                log_perf(
                    "portrait.resolve",
                    started_at,
                    person_name=person_name,
                    portrait_search_query=search_query,
                    resolved_from_query="wikipedia_page_exact_or_search",
                )
                return image_source

            for candidate_query in build_portrait_search_candidates(
                person_name,
                search_query,
                portrait_context,
            ):
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
                    image_source = strip_url_query_and_fragment(
                        str(image_meta.get("thumburl", "")).strip()
                    )
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
                        validate_image_url_reachable(
                            image_source,
                            person_name,
                            "Resolved portrait",
                        )
                    except Exception as exc:
                        if is_rate_limited_error(exc):
                            raise
                        continue

                    remember_successful_portrait_resolution(cache_key, image_source)
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
            attempted_queries_for_log = attempted_queries or attempted_queries_for_log
            if is_rate_limited_error(exc):
                last_rate_limit_error = f"portrait_search_query resolution failed for {person_name}: {exc}"
                if rate_limit_attempt + 1 < max_attempts:
                    backoff_seconds = portrait_retry_backoff_seconds(rate_limit_attempt)
                    logger.warning(
                        "Wikimedia portrait resolution rate-limited for %s; retrying in %.2fs (%s/%s): %s",
                        person_name,
                        backoff_seconds,
                        rate_limit_attempt + 1,
                        max_attempts,
                        exc,
                    )
                    if backoff_seconds > 0:
                        time.sleep(backoff_seconds)
                    continue
                resolution_error = last_rate_limit_error
                break

            resolution_error = (
                f"{resolution_error} ({exc})"
                if attempted_queries
                else f"portrait_search_query resolution failed for {person_name}: {exc}"
            )
            break
        else:
            if attempted_queries:
                attempted_queries_for_log = attempted_queries
            resolution_error = (
                f"No valid Wikimedia portrait found for {person_name} using portrait_search_query '{search_query}'"
            )
            break

    if last_rate_limit_error is not None and PORTRAIT_FALLBACK_MODE != "require_real":
        fallback_url = build_local_portrait_fallback_url(person_name)
        append_gemini_audit_log(
            "portrait.resolve",
            status="accepted_fallback",
            person_name=person_name,
            portrait_search_query=search_query,
            attempted_queries=attempted_queries_for_log,
            resolved_portrait_url=fallback_url,
            resolved_from_query="local_avatar_fallback",
            error=last_rate_limit_error,
        )
        logger.warning(
            "Using local avatar fallback for %s after Wikimedia throttling: %s",
            person_name,
            last_rate_limit_error,
        )
        return fallback_url

    with portrait_resolution_lock:
        portrait_resolution_cache[cache_key] = (False, resolution_error)
    append_gemini_audit_log(
        "portrait.resolve",
        status="rejected",
        person_name=person_name,
        portrait_search_query=search_query,
        attempted_queries=attempted_queries_for_log,
        error=resolution_error,
    )
    raise ValueError(resolution_error)


def inject_portrait_url(fragment: str, portrait_url: str) -> str:
    return fragment.replace(PORTRAIT_IMAGE_PLACEHOLDER, portrait_url)


def build_local_portrait_fallback_url(person_name: str) -> str:
    return f"{app_path('/portrait-fallback')}?name={quote(person_name)}"


def is_local_portrait_fallback_url(image_source: str) -> bool:
    parsed = urlsplit(image_source)
    if parsed.scheme or parsed.netloc:
        return False
    return parsed.path == app_path("/portrait-fallback")


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
        if is_local_portrait_fallback_url(image_source):
            continue
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
    guessing_ui_html = normalize_generated_html_fragment(round_item.guessing_ui_html)
    reveal_ui_html = normalize_generated_html_fragment(round_item.reveal_ui_html)

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
    portrait_context = extract_visible_text(guessing_ui_html)
    portrait_url = resolve_wikimedia_portrait_url(
        person_name,
        portrait_search_query,
        portrait_context,
    )
    guessing_ui_html = inject_portrait_url(guessing_ui_html, portrait_url)
    reveal_ui_html = inject_portrait_url(reveal_ui_html, portrait_url)
    reveal_ui_html = humanize_reveal_dates(
        reveal_ui_html,
        round_item.date_of_birth,
        round_item.date_of_death,
    )
    next_round_label = sanitize_next_round_label(round_item.next_round_label)
    reveal_ui_html = rewrite_next_round_control(
        reveal_ui_html,
        next_round_label,
        hide_on_mobile=True,
    )
    guessing_ui_html = wrap_round_fragment(guessing_ui_html)
    reveal_ui_html = wrap_round_fragment(reveal_ui_html)
    validate_embedded_image_urls(guessing_ui_html, person_name, "Guessing", require_image=True)
    validate_embedded_image_urls(reveal_ui_html, person_name, "Reveal")

    return {
        "person_name": person_name,
        "actual_status": round_item.actual_status,
        "date_of_birth": round_item.date_of_birth,
        "date_of_death": round_item.date_of_death,
        "next_round_label": next_round_label,
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
                temperature=ROUND_UI_TEMPERATURE,
                topP=ROUND_UI_TOP_P,
                candidateCount=1,
                maxOutputTokens=ROUND_UI_MAX_OUTPUT_TOKENS,
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

    try:
        round_item = GeneratedRound.model_validate_json(response_text)
    except Exception:
        payload = extract_json(response_text)
        if payload is None:
            raise
        if isinstance(payload, list):
            payload = payload[0]
        round_item = GeneratedRound.model_validate(payload)
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
                temperature=CANDIDATE_SELECTION_TEMPERATURE,
                topP=CANDIDATE_SELECTION_TOP_P,
                candidateCount=1,
                maxOutputTokens=CANDIDATE_SELECTION_MAX_OUTPUT_TOKENS,
                responseMimeType="application/json",
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

    try:
        selection = CandidateSelection.model_validate_json(response_text)
    except Exception:
        payload = extract_json(response_text)
        if payload is None:
            raise
        selection = CandidateSelection.model_validate(payload)
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
                "temperature": ROUND_UI_TEMPERATURE,
                "top_p": ROUND_UI_TOP_P,
                "max_output_tokens": ROUND_UI_MAX_OUTPUT_TOKENS,
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
                "temperature": CANDIDATE_SELECTION_TEMPERATURE,
                "top_p": CANDIDATE_SELECTION_TOP_P,
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
        date_of_birth=normalized["date_of_birth"],
        date_of_death=normalized["date_of_death"],
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


def choose_allowed_candidates(
    selection: CandidateSelection,
    forbidden: list[str],
    limit: int,
) -> tuple[list[LockedCandidate], list[str]]:
    forbidden_keys = {normalize_name_key(name) for name in forbidden}
    seen_keys: set[str] = set()
    accepted: list[LockedCandidate] = []
    rejected_names: list[str] = []

    for candidate in selection.candidates:
        if len(accepted) >= limit:
            break
        person_name = " ".join(candidate.person_name.split())
        key = normalize_name_key(person_name)
        if not person_name:
            continue
        if key in seen_keys or key in forbidden_keys:
            rejected_names.append(person_name)
            continue
        seen_keys.add(key)
        accepted.append(
            LockedCandidate(
                person_name=person_name,
                actual_status=candidate.actual_status,
                source="candidate_selection",
            )
        )

    return accepted, rejected_names


def choose_reused_candidates_from_history(
    ip_history: list[str],
    selected: list[LockedCandidate],
    limit: int,
) -> list[LockedCandidate]:
    selected_keys = {normalize_name_key(candidate.person_name) for candidate in selected}
    reused: list[LockedCandidate] = []

    for person_name in ip_history:
        if len(reused) >= limit:
            break
        key = normalize_name_key(person_name)
        if key in selected_keys:
            continue
        selected_keys.add(key)
        reused.append(
            LockedCandidate(
                person_name=person_name,
                actual_status=None,
                source="ip_history_reuse",
            )
        )

    return reused


def negotiate_round_candidates_sync(
    forbidden: list[str],
    ip_history: list[str],
    model_candidates: list[str],
    pool_size: int,
    audit_meta: dict[str, object],
) -> tuple[str, list[LockedCandidate]]:
    client_ip_key = normalize_client_ip(str(audit_meta.get("client_ip_key", "unknown")))
    working_forbidden = list(dict.fromkeys(forbidden))
    retry_notes: list[str] = []
    errors: list[str] = []
    approved: list[LockedCandidate] = []
    use_ip_pool = client_ip_key != "unknown"
    if use_ip_pool:
        approved = select_candidates_from_ip_pool(
            client_ip_key,
            working_forbidden,
            pool_size,
        )
    last_successful_model: str | None = None

    if approved:
        working_forbidden = merge_names_recency_preserving(
            working_forbidden,
            [candidate.person_name for candidate in approved],
        )
        append_gemini_audit_log(
            "candidate_bank.reuse",
            status="accepted",
            accepted_count=len(approved),
            person_names=[candidate.person_name for candidate in approved],
            source="ip_candidate_pool",
            **audit_meta,
        )
        if len(approved) >= pool_size:
            return "ip_candidate_pool", approved

    bank_candidates = select_candidates_from_bank(
        merge_names_recency_preserving(
            working_forbidden,
            [candidate.person_name for candidate in approved],
        ),
        pool_size - len(approved),
    )
    if bank_candidates:
        approved.extend(bank_candidates)
        if use_ip_pool:
            remember_candidates_for_ip_pool(client_ip_key, bank_candidates)
        working_forbidden = merge_names_recency_preserving(
            working_forbidden,
            [candidate.person_name for candidate in bank_candidates],
        )
        append_gemini_audit_log(
            "candidate_bank.reuse",
            status="accepted",
            accepted_count=len(bank_candidates),
            person_names=[candidate.person_name for candidate in bank_candidates],
            source="candidate_bank",
            **audit_meta,
        )
        if len(approved) >= pool_size:
            return "candidate_bank", approved

    for attempt in range(SESSION_NEGOTIATION_ATTEMPTS):
        if len(approved) >= pool_size:
            break

        remaining = pool_size - len(approved)
        requested_candidate_count = min(
            CANDIDATE_SELECTION_COUNT,
            max(CANDIDATE_SELECTION_MIN_COUNT, remaining * 5),
        )
        negotiation_forbidden = merge_names_recency_preserving(
            working_forbidden,
            [candidate.person_name for candidate in approved],
        )
        prompt = build_candidate_selection_prompt(
            negotiation_forbidden,
            requested_candidate_count,
            retry_notes,
        )
        accepted_this_attempt = 0

        for model_name in model_candidates:
            try:
                if modern_genai is not None:
                    selection = select_candidates_with_modern_sdk(model_name, prompt, audit_meta)
                else:
                    selection = select_candidates_with_legacy_sdk(model_name, prompt, audit_meta)
                candidates, rejected_names = choose_allowed_candidates(
                    selection,
                    negotiation_forbidden,
                    remaining,
                )
                working_forbidden = merge_names_recency_preserving(
                    working_forbidden,
                    rejected_names,
                )
                if not candidates:
                    rejected_tail = ", ".join(rejected_names[:12]) if rejected_names else "none"
                    raise ValueError(
                        "Candidate negotiation returned no newly allowed people. "
                        f"Rejected candidates: {rejected_tail}"
                    )

                approved.extend(candidates)
                working_forbidden = merge_names_recency_preserving(
                    working_forbidden,
                    [candidate.person_name for candidate in candidates],
                )
                accepted_this_attempt += len(candidates)
                last_successful_model = model_name
                remember_candidates_in_bank(candidates)
                if use_ip_pool:
                    remember_candidates_for_ip_pool(client_ip_key, candidates)
                append_gemini_audit_log(
                    "gemini.candidate_selection_result",
                    model=model_name,
                    status="accepted",
                    accepted_count=len(candidates),
                    person_names=[candidate.person_name for candidate in candidates],
                    **audit_meta,
                )
                if len(approved) >= pool_size:
                    return last_successful_model, approved
            except Exception as exc:
                append_gemini_audit_log(
                    "gemini.candidate_selection_result",
                    model=model_name,
                    status="rejected",
                    error=str(exc),
                    **audit_meta,
                )
                logger.warning("Candidate negotiation failed for %s: %s", model_name, exc)
                errors.append(f"{model_name} negotiation attempt {attempt + 1}: {exc}")
                retry_notes.append(build_retry_note(str(exc)))
                if "Rejected candidates:" in str(exc):
                    rejected_tail = str(exc).split("Rejected candidates:", 1)[1]
                    rejected_names = [
                        name.strip()
                        for name in rejected_tail.split(",")
                        if name.strip() and name.strip().lower() != "none"
                    ]
                    working_forbidden = merge_names_recency_preserving(
                        working_forbidden,
                        rejected_names,
                    )

        remaining = pool_size - len(approved)
        if remaining <= 0:
            break
        if accepted_this_attempt == 0:
            retry_notes.append(
                "Your previous candidate batch still reused forbidden or duplicate people. "
                "Go deeper and return a more diverse set of definitely different celebrities."
            )
        retry_notes.append(
            f"You still need to return {remaining} more unique allowed celebrities that are not in the forbidden list."
        )
        retry_notes = list(dict.fromkeys(retry_notes))

    if len(approved) < pool_size:
        reused = choose_reused_candidates_from_history(
            ip_history,
            approved,
            pool_size - len(approved),
        )
        if reused:
            approved.extend(reused)
            append_gemini_audit_log(
                "gemini.candidate_selection_reuse",
                status="used",
                person_names=[candidate.person_name for candidate in reused],
                reused_count=len(reused),
                **audit_meta,
            )

    if len(approved) >= pool_size:
        return last_successful_model or "ip_history_reuse", approved

    error_tail = " | ".join(errors) if errors else "insufficient approved candidates"
    raise RuntimeError(
        "Unable to negotiate enough round candidates for this IP. " + error_tail
    )


def generate_single_round_sync(
    forbidden: list[str],
    locked_candidate: LockedCandidate,
    model_candidates: list[str],
    race_models: bool = False,
) -> tuple[str, dict]:
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

    errors = []
    working_forbidden = [
        name
        for name in list(dict.fromkeys(forbidden))
        if normalize_name_key(name) != normalize_name_key(locked_candidate.person_name)
    ]
    retry_notes: list[str] = []

    for attempt in range(ROUND_GENERATION_ATTEMPTS):
        audit_request_id = str(uuid.uuid4())
        audit_meta = {
            "attempt": attempt + 1,
            "audit_request_id": audit_request_id,
        }
        prompt = build_generation_prompt(
            trim_retry_notes(retry_notes),
            locked_person_name=locked_candidate.person_name,
            locked_actual_status=locked_candidate.actual_status,
        )

        if race_models and len(model_candidates) > 1:
            forbidden_names_to_add = []
            attempt_retry_notes = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(model_candidates)
            ) as executor:
                future_to_model = {
                    executor.submit(
                        generate_round_candidate,
                        model_name,
                        prompt,
                        working_forbidden,
                        audit_meta | {"race_mode": True},
                        locked_candidate.person_name,
                        locked_candidate.actual_status,
                    ): model_name
                    for model_name in model_candidates
                }
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        return future.result()
                    except Exception as exc:
                        if should_skip_candidate_on_portrait_failure(str(exc)):
                            raise CandidatePortraitUnavailableError(str(exc)) from exc
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

            working_forbidden.extend(forbidden_names_to_add)
            working_forbidden = list(dict.fromkeys(working_forbidden))
            retry_notes = list(dict.fromkeys(retry_notes + attempt_retry_notes))
            continue

        for model_name in model_candidates:
            try:
                return generate_round_candidate(
                    model_name,
                    prompt,
                    working_forbidden,
                    audit_meta | {"race_mode": False},
                    locked_candidate.person_name,
                    locked_candidate.actual_status,
                )
            except Exception as exc:
                if should_skip_candidate_on_portrait_failure(str(exc)):
                    raise CandidatePortraitUnavailableError(str(exc)) from exc
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
                retry_notes = trim_retry_notes(retry_notes)

    error_tail = " | ".join(errors)
    if should_skip_candidate_on_generation_failure(error_tail):
        raise CandidatePortraitUnavailableError(error_tail)
    raise RuntimeError("All configured Gemini models failed. " + error_tail)


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
            <button onclick='startGame()' class='rounded-full bg-white px-10 py-4 text-lg font-black uppercase tracking-[0.2em] text-black transition hover:scale-105 active:scale-95'>
                Play Again
            </button>
        </div>
    </div>
    """


def build_portrait_fallback_svg(person_name: str) -> str:
    display_name = " ".join(person_name.split())[:60] or "Unknown"
    name_tokens = [token for token in display_name.split() if token]
    initials = "".join(token[0].upper() for token in name_tokens[:2]) or "?"
    escaped_initials = html.escape(initials)
    escaped_name = html.escape(display_name)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" viewBox="0 0 600 600" role="img" aria-label="{escaped_name}">
  <defs>
    <linearGradient id="ring" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#60a5fa" />
      <stop offset="100%" stop-color="#c084fc" />
    </linearGradient>
    <radialGradient id="avatar" cx="35%" cy="30%" r="85%">
      <stop offset="0%" stop-color="#4338ca" />
      <stop offset="100%" stop-color="#312e81" />
    </radialGradient>
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="18" stdDeviation="26" flood-color="#0f172a" flood-opacity="0.28" />
    </filter>
  </defs>
  <g filter="url(#glow)">
    <circle cx="300" cy="300" r="212" fill="url(#avatar)" />
    <circle cx="300" cy="300" r="212" fill="none" stroke="url(#ring)" stroke-width="12" />
    <circle cx="300" cy="300" r="126" fill="rgba(255,255,255,0.08)" stroke="rgba(255,255,255,0.12)" stroke-width="3" />
  </g>
  <text x="300" y="332" text-anchor="middle" fill="#f8fafc" font-size="128" font-family="Arial, Helvetica, sans-serif" font-weight="700">{escaped_initials}</text>
</svg>"""


def collect_session_reserved_names(session: dict) -> list[str]:
    reserved_names: list[str] = []
    reserved_names.extend(session.get("reserved_names", []))
    reserved_names.extend(session.get("used_names", []))
    reserved_names.extend(session.get("excluded_candidates", []))
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


def get_session_round_candidate(session: dict, target_round_number: int) -> LockedCandidate:
    round_candidates = session.get("round_candidates", [])
    index = target_round_number - 1
    if index < 0 or index >= len(round_candidates):
        raise RuntimeError(f"No negotiated candidate available for round {target_round_number}")

    candidate_data = round_candidates[index]
    return LockedCandidate(
        person_name=str(candidate_data["person_name"]),
        actual_status=candidate_data.get("actual_status"),
        source=str(candidate_data.get("source", "candidate_selection")),
    )


def verify_locked_candidate_status(locked_candidate: LockedCandidate) -> LockedCandidate:
    verified_status = resolve_person_actual_status(locked_candidate.person_name)
    if not verified_status:
        return locked_candidate
    if locked_candidate.actual_status == verified_status:
        return locked_candidate
    logger.warning(
        "Corrected locked status for %s from %s to %s",
        locked_candidate.person_name,
        locked_candidate.actual_status,
        verified_status,
    )
    return LockedCandidate(
        person_name=locked_candidate.person_name,
        actual_status=verified_status,
        source=f"{locked_candidate.source}_status_verified",
    )


def build_round_generation_forbidden(
    session: dict,
    locked_candidate: LockedCandidate,
) -> list[str]:
    forbidden: list[str] = []
    forbidden.extend(get_ip_history(session.get("client_ip_key", "unknown")))
    forbidden.extend(collect_ip_reserved_names(session.get("client_ip_key", "unknown")))
    forbidden.extend(collect_session_reserved_names(session))
    forbidden = merge_names_recency_preserving([], forbidden)
    return [
        name
        for name in forbidden
        if normalize_name_key(name) != normalize_name_key(locked_candidate.person_name)
    ]


def build_candidate_negotiation_forbidden(session: dict) -> list[str]:
    client_ip_key = session.get("client_ip_key", "unknown")
    forbidden: list[str] = []
    forbidden.extend(get_ip_history(client_ip_key))
    forbidden.extend(collect_ip_reserved_names(client_ip_key))
    forbidden.extend(collect_session_reserved_names(session))
    return merge_names_recency_preserving([], forbidden)


def should_skip_candidate_on_portrait_failure(error_message: str) -> bool:
    lowered = error_message.lower()
    return (
        "no valid wikimedia portrait found" in lowered
        or "portrait_search_query resolution failed" in lowered
    )


def should_skip_candidate_on_generation_failure(error_message: str) -> bool:
    lowered = error_message.lower()
    return (
        should_skip_candidate_on_portrait_failure(error_message)
        or "guessing ui places guess buttons over the portrait" in lowered
        or "guessing ui is missing guess buttons" in lowered
        or "reveal ui is missing next-round button" in lowered
        or "guessing ui includes spoiler wording" in lowered
        or "unsupported tailwind class" in lowered
        or "portrait placeholder" in lowered
        or "old image placeholder" in lowered
        or "returned non-json output" in lowered
        or "returned non-json candidate-selection output" in lowered
        or "json extraction failed" in lowered
        or "model returned forbidden person:" in lowered
        or "locked person" in lowered
        or "locked status" in lowered
        or "date_of_birth" in lowered
        or "date_of_death" in lowered
    )


def exclude_round_candidate(
    session: dict,
    target_round_number: int,
    locked_candidate: LockedCandidate,
    reason: str,
) -> None:
    candidate_index = target_round_number - 1
    round_candidates = session.get("round_candidates", [])
    if 0 <= candidate_index < len(round_candidates):
        round_candidates.pop(candidate_index)

    session["reserved_names"] = [
        name
        for name in session.get("reserved_names", [])
        if normalize_name_key(name) != normalize_name_key(locked_candidate.person_name)
    ]
    session["excluded_candidates"] = merge_names_recency_preserving(
        session.get("excluded_candidates", []),
        [locked_candidate.person_name],
    )
    forget_candidate_from_ip_pool(session.get("client_ip_key", "unknown"), locked_candidate.person_name)
    forget_candidate_from_bank(locked_candidate.person_name)
    append_gemini_audit_log(
        "round.candidate_excluded",
        status="excluded",
        person_name=locked_candidate.person_name,
        reason=reason,
        session_id=session.get("session_id"),
        target_round_number=target_round_number,
    )


async def generate_round_payload(
    session: dict,
    target_round_number: int,
    model_candidates: list[str],
    race_models: bool = False,
) -> tuple[str, dict]:
    portrait_skip_count = 0

    while True:
        if len(session.get("round_candidates", [])) < target_round_number:
            await top_up_session_candidates(session.get("session_id"))
        if len(session.get("round_candidates", [])) < target_round_number:
            raise RuntimeError(
                f"No negotiated candidate available for round {target_round_number}"
            )

        locked_candidate = get_session_round_candidate(session, target_round_number)
        if STATUS_VALIDATION_ENABLED:
            append_gemini_audit_log(
                "round.status_validation",
                status="enabled",
                person_name=locked_candidate.person_name,
                proposed_status=locked_candidate.actual_status,
                target_round_number=target_round_number,
                session_id=session.get("session_id"),
            )
            verified_candidate = await asyncio.to_thread(
                verify_locked_candidate_status,
                locked_candidate,
            )
            if verified_candidate != locked_candidate:
                session["round_candidates"][target_round_number - 1]["actual_status"] = (
                    verified_candidate.actual_status
                )
                session["round_candidates"][target_round_number - 1]["source"] = (
                    verified_candidate.source
                )
                locked_candidate = verified_candidate
        else:
            append_gemini_audit_log(
                "round.status_validation",
                status="disabled",
                person_name=locked_candidate.person_name,
                proposed_status=locked_candidate.actual_status,
                target_round_number=target_round_number,
                session_id=session.get("session_id"),
            )
            if locked_candidate.actual_status is not None:
                locked_candidate = LockedCandidate(
                    person_name=locked_candidate.person_name,
                    actual_status=None,
                    source=f"{locked_candidate.source}_status_unlocked",
                )

        forbidden = build_round_generation_forbidden(session, locked_candidate)
        try:
            model_name, round_data = await asyncio.to_thread(
                generate_single_round_sync,
                forbidden,
                locked_candidate,
                model_candidates,
                race_models,
            )
        except CandidatePortraitUnavailableError as exc:
            portrait_skip_count += 1
            exclude_round_candidate(
                session,
                target_round_number,
                locked_candidate,
                str(exc),
            )
            logger.warning(
                "Skipping %s for round %s after portrait failure: %s",
                locked_candidate.person_name,
                target_round_number,
                exc,
            )
            schedule_candidate_topup(session.get("session_id"))
            if portrait_skip_count >= max(SESSION_CANDIDATE_TARGET_COUNT, 10):
                raise RuntimeError(
                    f"Unable to find a portraitable candidate for round {target_round_number}"
                ) from exc
            continue

        session["used_names"] = merge_names_recency_preserving(
            session.get("used_names", []),
            [round_data["person_name"]],
        )
        remember_candidates_in_bank(
            [
                LockedCandidate(
                    person_name=round_data["person_name"],
                    actual_status=round_data["actual_status"],
                    source="round_generation",
                )
            ]
        )
        session["round_candidates"][target_round_number - 1]["actual_status"] = round_data["actual_status"]
        runtime_state["last_model"] = model_name
        runtime_state["last_error"] = None
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
            if current_session["round_number"] + len(current_session["queued_rounds"]) >= ROUNDS_PER_GAME:
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


async def top_up_session_candidates(session_id: str) -> None:
    session = sessions.get(session_id)
    if not session:
        return

    client_ip_key = session.get("client_ip_key", "unknown")
    try:
        while True:
            current_session = sessions.get(session_id)
            if current_session is None:
                return

            needed = SESSION_CANDIDATE_TARGET_COUNT - len(current_session.get("round_candidates", []))
            if needed <= 0:
                return

            forbidden = build_candidate_negotiation_forbidden(current_session)
            ip_history = get_ip_history(client_ip_key)
            _, new_candidates = await asyncio.to_thread(
                negotiate_round_candidates_sync,
                forbidden,
                ip_history,
                BACKGROUND_MODEL_CANDIDATES,
                needed,
                {
                    "session_id": session_id,
                    "client_ip_key": client_ip_key,
                    "background_topup": True,
                },
            )

            current_session = sessions.get(session_id)
            if current_session is None:
                return
            if not new_candidates:
                return

            new_names = [candidate.person_name for candidate in new_candidates]
            current_session["reserved_names"] = merge_names_recency_preserving(
                current_session.get("reserved_names", []),
                new_names,
            )
            current_session["round_candidates"].extend(
                {
                    "person_name": candidate.person_name,
                    "actual_status": candidate.actual_status,
                    "source": candidate.source,
                }
                for candidate in new_candidates
            )
            current_session["candidate_topup_error"] = None
            logger.info(
                "Background candidate top-up added %s celebrities for session %s (%s total reserved)",
                len(new_candidates),
                session_id,
                len(current_session["round_candidates"]),
            )
    except Exception as exc:
        current_session = sessions.get(session_id)
        if current_session is not None:
            current_session["candidate_topup_error"] = str(exc)
        logger.warning("Candidate top-up failed for session %s: %s", session_id, exc)
    finally:
        current_session = sessions.get(session_id)
        if current_session is not None:
            current_session["candidate_topup_task"] = None


def schedule_prefetch(session_id: str) -> None:
    session = sessions.get(session_id)
    if not session:
        return
    if session["round_number"] >= ROUNDS_PER_GAME:
        return
    if len(session.get("queued_rounds", [])) >= PREFETCH_ROUND_BUFFER:
        return

    existing_task = session.get("prefetch_task")
    if existing_task is not None and not existing_task.done():
        return

    session["prefetch_task"] = asyncio.create_task(prefetch_next_rounds(session_id))


def schedule_candidate_topup(session_id: str) -> None:
    session = sessions.get(session_id)
    if not session:
        return
    if len(session.get("round_candidates", [])) >= SESSION_CANDIDATE_TARGET_COUNT:
        return

    existing_task = session.get("candidate_topup_task")
    if existing_task is not None and not existing_task.done():
        return

    session["candidate_topup_task"] = asyncio.create_task(top_up_session_candidates(session_id))


async def get_next_round_for_session(session_id: str, session: dict) -> dict:
    queued_rounds = session.get("queued_rounds", [])
    if queued_rounds:
        queued_round = queued_rounds.pop(0)
        session["active_round"] = queued_round
        return queued_round

    existing_task = session.get("prefetch_task")
    if existing_task is not None and not existing_task.done():
        while True:
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


app = FastAPI(title="Alive or Dead POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index():
    start_session_url = app_path("/api/start-session")
    next_round_url = app_path("/api/next-round")
    guess_url = app_path("/api/guess")
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
            .round-fragment {{
                width: 100%;
            }}
            .round-fragment > * {{
                width: 100% !important;
                max-width: 100% !important;
                box-sizing: border-box;
                margin-left: auto;
                margin-right: auto;
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
                        Gemini designs each round on demand, then the backend resolves and verifies a matching Wikimedia portrait before the round is shown.
                    </p>
                    <div class="mt-8 flex flex-wrap items-center gap-4">
                        <button id="start-btn" onclick="startGame()" class="rounded-full bg-white px-12 py-5 text-lg font-black uppercase tracking-[0.24em] text-black transition hover:scale-105 active:scale-95">
                            Start Game
                        </button>
                        <div class="rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm uppercase tracking-[0.28em] text-zinc-300">
                            10 live rounds
                        </div>
                    </div>
                </div>
                <div class="relative">
                    <div class="absolute inset-0 rotate-[-7deg] rounded-[2.75rem] bg-gradient-to-br from-amber-400/35 via-rose-500/20 to-sky-400/20 blur-3xl"></div>
                    <div class="relative overflow-hidden rounded-[2.5rem] border border-white/10 bg-zinc-950/80 p-6 shadow-2xl">
                        <div class="flex items-center justify-between">
                            <p class="text-xs font-bold uppercase tracking-[0.42em] text-zinc-500">Generative Round</p>
                            <p class="rounded-full border border-white/10 px-3 py-1 text-[10px] font-bold uppercase tracking-[0.3em] text-amber-200">Live</p>
                        </div>
                        <div class="mt-6 grid gap-4">
                            <div class="rounded-[2rem] border border-white/10 bg-gradient-to-br from-white/10 to-white/0 p-5">
                                <p class="text-xs uppercase tracking-[0.35em] text-zinc-400">Design brief</p>
                                <p class="mt-3 text-2xl font-black uppercase tracking-tight">Poster-scale names. Tabloid tension. Verified Wikimedia portraits.</p>
                            </div>
                            <div class="grid grid-cols-2 gap-4">
                                <div class="rounded-[1.75rem] border border-amber-300/20 bg-amber-300/10 p-4">
                                    <p class="text-xs uppercase tracking-[0.3em] text-amber-200">Guess</p>
                                    <p class="mt-2 text-sm leading-relaxed text-zinc-200">Fresh AI-designed layout generated in real time.</p>
                                </div>
                                <div class="rounded-[1.75rem] border border-sky-300/20 bg-sky-300/10 p-4">
                                    <p class="text-xs uppercase tracking-[0.3em] text-sky-200">Image</p>
                                    <p class="mt-2 text-sm leading-relaxed text-zinc-200">Gemini suggests the portrait search query, then the backend resolves a live Wikimedia image URL.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <script>
            let sessionId = null;
            const START_SESSION_URL = {json.dumps(start_session_url)};
            const NEXT_ROUND_URL = {json.dumps(next_round_url)};
            const GUESS_URL = {json.dumps(guess_url)};

            function renderLoader(title, detail) {{
                document.getElementById('app').innerHTML = `
                    <div class="relative w-full max-w-4xl overflow-hidden rounded-[2.75rem] border border-white/10 bg-zinc-950/95 p-10 text-center shadow-2xl">
                        <div class="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-amber-300 via-rose-400 to-sky-400"></div>
                        <div class="mx-auto mb-8 flex h-24 w-24 items-center justify-center rounded-full border border-white/15 bg-white/5">
                            <div class="h-16 w-16 rounded-full border-[6px] border-white border-t-transparent animate-spin"></div>
                        </div>
                        <p class="text-xs font-bold uppercase tracking-[0.42em] text-zinc-500">Live generation</p>
                        <h2 id="status-title" class="mt-4 text-3xl font-black uppercase tracking-[0.18em] md:text-4xl"></h2>
                        <p id="status-detail" class="mx-auto mt-5 max-w-2xl text-lg leading-relaxed text-zinc-400"></p>
                        <button onclick="location.reload()" class="mt-8 rounded-full border border-white/15 px-6 py-3 text-sm font-bold uppercase tracking-[0.2em] text-zinc-200 transition hover:border-white/40 hover:text-white">
                            Reload
                        </button>
                    </div>`;
                document.getElementById('status-title').textContent = title;
                document.getElementById('status-detail').textContent = detail;
            }}

            function renderFatal(message) {{
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
                sessionId = null;
                renderLoader(
                    'Identifying Your Celebrities',
                    'The game is negotiating your celebrity pool and building round one. The game will start momentarily.'
                );

                try {{
                    const response = await fetch(START_SESSION_URL, {{ method: 'POST' }});
                    const data = await response.json();
                    if (!response.ok) {{
                        renderFatal(data.detail || 'Unable to start a new session.');
                        return;
                    }}
                    sessionId = data.session_id;
                    document.getElementById('app').innerHTML = data.guessing_ui_html;
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
                }} catch (error) {{
                    renderFatal('The next round failed to load.');
                }}
            }}

            async function submitGuess(guess) {{
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
        "status_validation_enabled": STATUS_VALIDATION_ENABLED,
        "gemini_audit_log_enabled": GEMINI_AUDIT_LOG_ENABLED,
        "gemini_audit_log_file": str(GEMINI_AUDIT_LOG_FILE),
        "image_url_validation_enabled": IMAGE_URL_VALIDATION_ENABLED,
        "wikimedia_search_timeout_seconds": WIKIMEDIA_SEARCH_TIMEOUT_SECONDS,
        "portrait_fallback_mode": PORTRAIT_FALLBACK_MODE,
        "wikimedia_429_retry_count": WIKIMEDIA_429_RETRY_COUNT,
        "wikimedia_429_retry_backoff_ms": WIKIMEDIA_429_RETRY_BACKOFF_MS,
        "repeat_history_scope": "per_ip_fifo",
        "ip_history_max_names": IP_HISTORY_MAX_NAMES,
        "session_candidate_count": SESSION_APPROVED_POOL_SIZE,
        "session_candidate_target_count": SESSION_CANDIDATE_TARGET_COUNT,
        "candidate_ip_pool_file": str(CANDIDATE_IP_POOL_FILE),
        "candidate_ip_pool_max_entries": IP_CANDIDATE_POOL_MAX_ENTRIES,
        "candidate_ip_pool_ttl_seconds": IP_CANDIDATE_POOL_TTL_SECONDS,
        "candidate_bank_file": str(CANDIDATE_BANK_FILE),
        "candidate_bank_max_entries": CANDIDATE_BANK_MAX_ENTRIES,
        "candidate_bank_ttl_seconds": CANDIDATE_BANK_TTL_SECONDS,
        "wikimedia_min_request_interval_ms": WIKIMEDIA_MIN_REQUEST_INTERVAL_MS,
        "image_contract": "backend_wikimedia_query_resolution",
    }


@app.get("/portrait-fallback")
async def portrait_fallback(name: str = "Unknown"):
    return Response(
        content=build_portrait_fallback_svg(name),
        media_type="image/svg+xml",
    )


@app.post("/api/start-session")
async def start_session(request: FastAPIRequest):
    session_id = str(uuid.uuid4())
    client_ip_key = normalize_client_ip(extract_client_ip(request))
    ip_history = get_ip_history(client_ip_key)
    forbidden = merge_names_recency_preserving(
        ip_history,
        collect_ip_reserved_names(client_ip_key),
    )

    try:
        _, round_candidates = await asyncio.to_thread(
            negotiate_round_candidates_sync,
            forbidden,
            ip_history,
            FAST_MODEL_CANDIDATES,
            max(SESSION_APPROVED_POOL_SIZE, ROUNDS_PER_GAME),
            {
                "session_id": session_id,
                "client_ip_key": client_ip_key,
            },
        )
        reserved_names = [candidate.person_name for candidate in round_candidates]
        session = {
            "session_id": session_id,
            "client_ip_key": client_ip_key,
            "score": 0,
            "round_number": 0,
            "history": [],
            "used_names": [],
            "reserved_names": reserved_names,
            "excluded_candidates": [],
            "round_candidates": [
                {
                    "person_name": candidate.person_name,
                    "actual_status": candidate.actual_status,
                    "source": candidate.source,
                }
                for candidate in round_candidates
            ],
            "active_round": None,
            "queued_rounds": [],
            "prefetch_task": None,
            "prefetch_error": None,
            "candidate_topup_task": None,
            "candidate_topup_error": None,
        }
        sessions[session_id] = session
        round_data = await generate_round_for_session(
            session,
            1,
            FAST_MODEL_CANDIDATES,
            RACE_FAST_MODELS,
        )
    except Exception as exc:
        runtime_state["last_error"] = str(exc)
        sessions.pop(session_id, None)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    session["round_number"] = 1
    update_ip_history(client_ip_key, [round_data["person_name"]])
    schedule_prefetch(session_id)
    schedule_candidate_topup(session_id)
    return {
        "session_id": session_id,
        "status": "ready",
        "guessing_ui_html": round_data["guessing_ui_html"],
    }


@app.get("/api/next-round")
async def next_round(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    if session["round_number"] >= ROUNDS_PER_GAME:
        return {"status": "finale", "finale_html": render_finale_html(session)}

    try:
        round_data = await get_next_round_for_session(session_id, session)
    except Exception as exc:
        runtime_state["last_error"] = str(exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    session["round_number"] += 1
    update_ip_history(session.get("client_ip_key", "unknown"), [round_data["person_name"]])
    schedule_prefetch(session_id)
    schedule_candidate_topup(session_id)
    return {"status": "playing", "guessing_ui_html": round_data["guessing_ui_html"]}


@app.post("/api/guess")
async def guess(request: FastAPIRequest):
    data = await request.json()
    session_id = data.get("session_id")
    user_guess = str(data.get("guess", "")).lower()

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if user_guess not in {"alive", "dead"}:
        raise HTTPException(status_code=400, detail="Guess must be 'alive' or 'dead'")

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

    color = "text-emerald-400" if is_correct else "text-rose-400"
    text = "CORRECT" if is_correct else "INCORRECT"
    injection = (
        "<div class='pointer-events-none fixed left-0 right-0 top-8 z-50 flex justify-center'>"
        "<div class='rounded-full border border-white/10 bg-black/90 px-8 py-3 shadow-2xl'>"
        f"<span class='{color} text-2xl font-black uppercase tracking-[0.25em]'>{text}</span>"
        "</div>"
        "</div>"
    )
    return {
        "reveal_ui_html": injection
        + round_data["reveal_ui_html"]
        + build_mobile_next_round_bar(round_data.get("next_round_label", "Next Round"))
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
