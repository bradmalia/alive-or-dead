"""Durable, transactional inventory for pre-validated game rounds."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import unicodedata
from typing import Callable


SCHEMA_VERSION = 4
ROUND_STATES = ("available", "packed", "reserved", "consumed", "retired")
PACK_STATES = ("available", "reserved", "consumed", "retired")
JOB_STATES = ("queued", "leased", "completed", "failed")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None = None) -> str:
    current = value or utc_now()
    return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_category_key(category: str) -> str:
    normalized = unicodedata.normalize("NFKC", category)
    return " ".join(normalized.split()).casefold()


def normalize_person_key(person_name: str) -> str:
    normalized = unicodedata.normalize("NFKC", person_name)
    return " ".join(normalized.split()).casefold()


def connect(db_path: Path | str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Apply additive inventory schema migrations to an open database."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS app_schema_migrations (
            component TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS round_inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_key TEXT NOT NULL,
            category TEXT NOT NULL,
            person_key TEXT NOT NULL,
            person_name TEXT NOT NULL,
            actual_status TEXT NOT NULL CHECK (actual_status IN ('alive', 'dead')),
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            payload_version INTEGER NOT NULL DEFAULT 1,
            source_model TEXT,
            wikidata_entity_id TEXT,
            wikipedia_title TEXT,
            birth_date TEXT,
            portrait_url TEXT,
            state TEXT NOT NULL DEFAULT 'available'
                CHECK (state IN ('available', 'packed', 'reserved', 'consumed', 'retired')),
            reserved_by TEXT,
            validated_at TEXT NOT NULL,
            validation_attempted_at TEXT,
            validation_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            expires_at TEXT,
            UNIQUE (category_key, person_key)
        );

        CREATE INDEX IF NOT EXISTS round_inventory_available_lookup
            ON round_inventory(category_key, state, created_at, id);

        CREATE TABLE IF NOT EXISTS game_packs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_key TEXT NOT NULL,
            category TEXT NOT NULL,
            mode TEXT NOT NULL CHECK (mode IN ('classic')),
            round_count INTEGER NOT NULL CHECK (round_count > 0),
            state TEXT NOT NULL DEFAULT 'available'
                CHECK (state IN ('available', 'reserved', 'consumed', 'retired')),
            reserved_by TEXT,
            reserved_at TEXT,
            reservation_expires_at TEXT,
            source_job_id INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS game_packs_available_lookup
            ON game_packs(category_key, mode, state, round_count, created_at, id);
        CREATE UNIQUE INDEX IF NOT EXISTS game_packs_reserved_session_unique
            ON game_packs(reserved_by) WHERE reserved_by IS NOT NULL;

        CREATE TABLE IF NOT EXISTS game_pack_rounds (
            pack_id INTEGER NOT NULL REFERENCES game_packs(id) ON DELETE CASCADE,
            round_id INTEGER NOT NULL UNIQUE
                REFERENCES round_inventory(id) ON DELETE RESTRICT,
            position INTEGER NOT NULL CHECK (position > 0),
            PRIMARY KEY (pack_id, position)
        );

        CREATE TABLE IF NOT EXISTS inventory_refill_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_key TEXT NOT NULL,
            category TEXT NOT NULL,
            requested_packs INTEGER NOT NULL CHECK (requested_packs > 0),
            rounds_per_pack INTEGER NOT NULL CHECK (rounds_per_pack > 0),
            alive_per_pack INTEGER NOT NULL CHECK (
                alive_per_pack >= 0 AND alive_per_pack <= rounds_per_pack
            ),
            state TEXT NOT NULL DEFAULT 'queued'
                CHECK (state IN ('queued', 'leased', 'completed', 'failed')),
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
            lease_owner TEXT,
            lease_expires_at TEXT,
            not_before TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS inventory_refill_jobs_active_category
            ON inventory_refill_jobs(category_key)
            WHERE state IN ('queued', 'leased');
        CREATE INDEX IF NOT EXISTS inventory_refill_jobs_claimable
            ON inventory_refill_jobs(state, not_before, created_at, id);
        """
    )
    pack_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(game_packs)")
    }
    if "source_job_id" not in pack_columns:
        connection.execute("ALTER TABLE game_packs ADD COLUMN source_job_id INTEGER")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS game_packs_source_job "
        "ON game_packs(source_job_id, state)"
    )
    round_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(round_inventory)")
    }
    if "validated_at" not in round_columns:
        connection.execute("ALTER TABLE round_inventory ADD COLUMN validated_at TEXT")
    if "validation_attempted_at" not in round_columns:
        connection.execute("ALTER TABLE round_inventory ADD COLUMN validation_attempted_at TEXT")
    if "validation_error" not in round_columns:
        connection.execute("ALTER TABLE round_inventory ADD COLUMN validation_error TEXT")
    connection.execute(
        "UPDATE round_inventory SET validated_at = created_at WHERE validated_at IS NULL"
    )
    connection.execute(
        """
        INSERT INTO app_schema_migrations(component, version, applied_at)
        VALUES ('round_inventory', ?, ?)
        ON CONFLICT(component) DO UPDATE SET
            version = excluded.version,
            applied_at = CASE
                WHEN app_schema_migrations.version < excluded.version
                THEN excluded.applied_at
                ELSE app_schema_migrations.applied_at
            END
        """,
        (SCHEMA_VERSION, utc_iso()),
    )


def _job_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in row.keys()}


def enqueue_refill_job(
    db_path: Path | str,
    category: str,
    *,
    requested_packs: int,
    rounds_per_pack: int = 10,
    alive_per_pack: int = 5,
    max_attempts: int = 3,
) -> dict:
    if requested_packs < 1:
        raise ValueError("requested_packs must be positive")
    if rounds_per_pack < 1:
        raise ValueError("rounds_per_pack must be positive")
    if not 0 <= alive_per_pack <= rounds_per_pack:
        raise ValueError("alive_per_pack must fall within the pack size")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    connection = connect(db_path)
    try:
        initialize_schema(connection)
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        category_key = normalize_category_key(category)
        existing = connection.execute(
            """
            SELECT * FROM inventory_refill_jobs
            WHERE category_key = ? AND state IN ('queued', 'leased')
            LIMIT 1
            """,
            (category_key,),
        ).fetchone()
        if existing is not None:
            connection.rollback()
            return _job_dict(existing) | {"created": False}
        now = utc_iso()
        cursor = connection.execute(
            """
            INSERT INTO inventory_refill_jobs (
                category_key, category, requested_packs, rounds_per_pack,
                alive_per_pack, state, attempt_count, max_attempts,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'queued', 0, ?, ?, ?)
            """,
            (
                category_key,
                " ".join(category.split()),
                requested_packs,
                rounds_per_pack,
                alive_per_pack,
                max_attempts,
                now,
                now,
            ),
        )
        job_id = int(cursor.lastrowid)
        row = connection.execute(
            "SELECT * FROM inventory_refill_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        connection.commit()
        return _job_dict(row) | {"created": True}
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _recover_expired_job_leases(connection: sqlite3.Connection, now: str) -> int:
    return connection.execute(
        """
        UPDATE inventory_refill_jobs
        SET state = 'queued', lease_owner = NULL, lease_expires_at = NULL,
            not_before = NULL, updated_at = ?
        WHERE state = 'leased' AND lease_expires_at IS NOT NULL
          AND datetime(lease_expires_at) <= datetime(?)
        """,
        (now, now),
    ).rowcount


def claim_refill_job(
    db_path: Path | str,
    worker_id: str,
    *,
    lease_ttl_seconds: int = 300,
) -> dict | None:
    if not worker_id.strip():
        raise ValueError("worker_id cannot be blank")
    if lease_ttl_seconds < 1:
        raise ValueError("lease_ttl_seconds must be positive")
    connection = connect(db_path)
    try:
        initialize_schema(connection)
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        now_value = utc_now()
        now = utc_iso(now_value)
        _recover_expired_job_leases(connection, now)
        row = connection.execute(
            """
            SELECT * FROM inventory_refill_jobs
            WHERE state = 'queued'
              AND (not_before IS NULL OR datetime(not_before) <= datetime(?))
            ORDER BY created_at, id
            LIMIT 1
            """,
            (now,),
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        expires_at = utc_iso(now_value + timedelta(seconds=lease_ttl_seconds))
        updated = connection.execute(
            """
            UPDATE inventory_refill_jobs
            SET state = 'leased', attempt_count = attempt_count + 1,
                lease_owner = ?, lease_expires_at = ?, updated_at = ?
            WHERE id = ? AND state = 'queued'
            """,
            (worker_id, expires_at, now, row["id"]),
        ).rowcount
        if updated != 1:
            connection.rollback()
            return None
        leased = connection.execute(
            "SELECT * FROM inventory_refill_jobs WHERE id = ?",
            (row["id"],),
        ).fetchone()
        connection.commit()
        return _job_dict(leased)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def renew_refill_job_lease(
    db_path: Path | str,
    job_id: int,
    worker_id: str,
    *,
    lease_ttl_seconds: int = 300,
) -> bool:
    if lease_ttl_seconds < 1:
        raise ValueError("lease_ttl_seconds must be positive")
    now_value = utc_now()
    with connect(db_path) as connection:
        updated = connection.execute(
            """
            UPDATE inventory_refill_jobs
            SET lease_expires_at = ?, updated_at = ?
            WHERE id = ? AND state = 'leased' AND lease_owner = ?
              AND datetime(lease_expires_at) > datetime(?)
            """,
            (
                utc_iso(now_value + timedelta(seconds=lease_ttl_seconds)),
                utc_iso(now_value),
                job_id,
                worker_id,
                utc_iso(now_value),
            ),
        ).rowcount
    return updated == 1


def complete_refill_job(db_path: Path | str, job_id: int, worker_id: str) -> bool:
    now = utc_iso()
    with connect(db_path) as connection:
        updated = connection.execute(
            """
            UPDATE inventory_refill_jobs
            SET state = 'completed', lease_owner = NULL, lease_expires_at = NULL,
                completed_at = ?, updated_at = ?
            WHERE id = ? AND state = 'leased' AND lease_owner = ?
            """,
            (now, now, job_id, worker_id),
        ).rowcount
    return updated == 1


def fail_refill_job(
    db_path: Path | str,
    job_id: int,
    worker_id: str,
    error: str,
    *,
    retry_delay_seconds: int = 30,
) -> str | None:
    connection = connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT attempt_count, max_attempts FROM inventory_refill_jobs
            WHERE id = ? AND state = 'leased' AND lease_owner = ?
            """,
            (job_id, worker_id),
        ).fetchone()
        if row is None:
            connection.rollback()
            return None
        exhausted = int(row["attempt_count"]) >= int(row["max_attempts"])
        next_state = "failed" if exhausted else "queued"
        now_value = utc_now()
        now = utc_iso(now_value)
        not_before = (
            None
            if exhausted
            else utc_iso(now_value + timedelta(seconds=max(0, retry_delay_seconds)))
        )
        connection.execute(
            """
            UPDATE inventory_refill_jobs
            SET state = ?, lease_owner = NULL, lease_expires_at = NULL,
                not_before = ?, last_error = ?, completed_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                next_state,
                not_before,
                error[:2000],
                now if exhausted else None,
                now,
                job_id,
            ),
        )
        connection.commit()
        return next_state
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def validate_round_payload(payload: dict) -> dict:
    required_fields = (
        "person_name",
        "actual_status",
        "guessing_ui_html",
        "reveal_ui_html",
    )
    missing = [field for field in required_fields if not str(payload.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Inventory round is missing required fields: {', '.join(missing)}")
    if payload["actual_status"] not in {"alive", "dead"}:
        raise ValueError("Inventory round actual_status must be alive or dead")
    return payload


def add_round(
    db_path: Path | str,
    category: str,
    payload: dict,
    *,
    source_model: str | None = None,
    expires_at: str | None = None,
) -> int:
    validate_round_payload(payload)
    normalized_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(normalized_payload.encode("utf-8")).hexdigest()
    now = utc_iso()
    portrait_url = payload.get("portrait_url")
    with connect(db_path) as connection:
        initialize_schema(connection)
        cursor = connection.execute(
            """
            INSERT INTO round_inventory (
                category_key, category, person_key, person_name, actual_status,
                payload_json, payload_sha256, source_model, wikidata_entity_id,
                wikipedia_title, birth_date, portrait_url, state, validated_at,
                created_at, updated_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'available', ?, ?, ?, ?)
            """,
            (
                normalize_category_key(category),
                " ".join(category.split()),
                normalize_person_key(payload["person_name"]),
                " ".join(payload["person_name"].split()),
                payload["actual_status"],
                normalized_payload,
                payload_hash,
                source_model,
                payload.get("wikidata_entity_id"),
                payload.get("wikipedia_title"),
                payload.get("birth_date"),
                portrait_url,
                now,
                now,
                now,
                expires_at,
            ),
        )
        return int(cursor.lastrowid)


def category_person_names(db_path: Path | str, category: str) -> list[str]:
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT person_name FROM round_inventory WHERE category_key = ? ORDER BY id",
            (normalize_category_key(category),),
        ).fetchall()
    return [str(row["person_name"]) for row in rows]


def create_classic_pack(
    db_path: Path | str,
    category: str,
    *,
    round_count: int = 10,
    alive_per_pack: int | None = None,
    source_job_id: int | None = None,
    max_round_age_seconds: int | None = None,
) -> int | None:
    """Atomically move available rounds into one ready Classic pack."""
    if round_count < 1:
        raise ValueError("round_count must be positive")
    if alive_per_pack is not None and not 0 <= alive_per_pack <= round_count:
        raise ValueError("alive_per_pack must fall within the pack size")
    if max_round_age_seconds is not None and max_round_age_seconds < 1:
        raise ValueError("max_round_age_seconds must be positive")
    connection = connect(db_path)
    try:
        initialize_schema(connection)
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        category_key = normalize_category_key(category)
        freshness_clause = ""
        query_parameters: list[object] = [category_key]
        if max_round_age_seconds is not None:
            freshness_clause = " AND datetime(validated_at) > datetime(?)"
            query_parameters.append(
                utc_iso(utc_now() - timedelta(seconds=max_round_age_seconds))
            )
        if alive_per_pack is None:
            rows = connection.execute(
                f"""
                SELECT id, actual_status FROM round_inventory
                WHERE category_key = ? AND state = 'available'
                  AND (expires_at IS NULL OR datetime(expires_at) > CURRENT_TIMESTAMP)
                  {freshness_clause}
                ORDER BY created_at, id
                LIMIT ?
                """,
                (*query_parameters, round_count),
            ).fetchall()
        else:
            selected_by_status: dict[str, list[sqlite3.Row]] = {}
            for status, required_count in (
                ("alive", alive_per_pack),
                ("dead", round_count - alive_per_pack),
            ):
                selected_by_status[status] = connection.execute(
                    f"""
                    SELECT id, actual_status FROM round_inventory
                    WHERE category_key = ? AND state = 'available' AND actual_status = ?
                      AND (expires_at IS NULL OR datetime(expires_at) > CURRENT_TIMESTAMP)
                      {freshness_clause}
                    ORDER BY created_at, id
                    LIMIT ?
                    """,
                    (category_key, status, *query_parameters[1:], required_count),
                ).fetchall()
            if any(
                len(selected_by_status[status]) != required_count
                for status, required_count in (
                    ("alive", alive_per_pack),
                    ("dead", round_count - alive_per_pack),
                )
            ):
                connection.rollback()
                return None
            rows = []
            alive_rows = list(selected_by_status["alive"])
            dead_rows = list(selected_by_status["dead"])
            while alive_rows or dead_rows:
                if alive_rows:
                    rows.append(alive_rows.pop(0))
                if dead_rows:
                    rows.append(dead_rows.pop(0))
        if len(rows) != round_count:
            connection.rollback()
            return None
        now = utc_iso()
        cursor = connection.execute(
            """
            INSERT INTO game_packs (
                category_key, category, mode, round_count, state, source_job_id,
                created_at, updated_at
            ) VALUES (?, ?, 'classic', ?, 'available', ?, ?, ?)
            """,
            (
                category_key,
                " ".join(category.split()),
                round_count,
                source_job_id,
                now,
                now,
            ),
        )
        pack_id = int(cursor.lastrowid)
        round_ids = [int(row["id"]) for row in rows]
        connection.executemany(
            "INSERT INTO game_pack_rounds(pack_id, round_id, position) VALUES (?, ?, ?)",
            [(pack_id, round_id, position) for position, round_id in enumerate(round_ids, 1)],
        )
        placeholders = ",".join("?" for _ in round_ids)
        updated = connection.execute(
            f"UPDATE round_inventory SET state = 'packed', updated_at = ? "
            f"WHERE state = 'available' AND id IN ({placeholders})",
            (now, *round_ids),
        ).rowcount
        if updated != round_count:
            raise RuntimeError("Inventory changed while a game pack was being assembled")
        connection.commit()
        return pack_id
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def ensure_minimum_pack_stock(
    db_path: Path | str,
    category: str,
    *,
    minimum_ready_packs: int,
    rounds_per_pack: int = 10,
    alive_per_pack: int = 5,
    max_attempts: int = 3,
    failed_job_cooldown_seconds: int = 3600,
) -> dict:
    if minimum_ready_packs < 0:
        raise ValueError("minimum_ready_packs cannot be negative")
    category_key = normalize_category_key(category)
    with connect(db_path) as connection:
        ready_packs = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM game_packs
                WHERE category_key = ? AND mode = 'classic' AND state = 'available'
                  AND round_count = ?
                """,
                (category_key, rounds_per_pack),
            ).fetchone()[0]
        )
        loose_rows = connection.execute(
            """
            SELECT actual_status, COUNT(*) AS count FROM round_inventory
            WHERE category_key = ? AND state = 'available'
              AND (expires_at IS NULL OR datetime(expires_at) > CURRENT_TIMESTAMP)
            GROUP BY actual_status
            """,
            (category_key,),
        ).fetchall()
        failed_job = connection.execute(
            """
            SELECT id, updated_at, last_error FROM inventory_refill_jobs
            WHERE category_key = ? AND state = 'failed'
              AND datetime(updated_at) > datetime(?)
            ORDER BY updated_at DESC, id DESC LIMIT 1
            """,
            (
                category_key,
                utc_iso(utc_now() - timedelta(seconds=max(0, failed_job_cooldown_seconds))),
            ),
        ).fetchone()
    loose_rounds = {"alive": 0, "dead": 0}
    loose_rounds.update({str(row["actual_status"]): int(row["count"]) for row in loose_rows})
    deficit = max(0, minimum_ready_packs - ready_packs)
    job = None
    if deficit and failed_job is None:
        job = enqueue_refill_job(
            db_path,
            category,
            requested_packs=deficit,
            rounds_per_pack=rounds_per_pack,
            alive_per_pack=alive_per_pack,
            max_attempts=max_attempts,
        )
    return {
        "category": " ".join(category.split()),
        "minimum_ready_packs": minimum_ready_packs,
        "ready_packs": ready_packs,
        "deficit_packs": deficit,
        "loose_rounds": loose_rounds,
        "blocked_by_failed_job": _job_dict(failed_job) if failed_job is not None else None,
        "job": job,
    }


def count_job_created_packs(db_path: Path | str, job_id: int) -> int:
    with connect(db_path) as connection:
        return int(
            connection.execute(
                """
                SELECT COUNT(*) FROM game_packs
                WHERE source_job_id = ? AND state != 'retired'
                """,
                (job_id,),
            ).fetchone()[0]
        )


RoundRevalidator = Callable[[dict], dict]


def revalidate_stale_available_rounds(
    db_path: Path | str,
    revalidate: RoundRevalidator,
    *,
    max_round_age_seconds: int,
    limit: int = 100,
) -> dict:
    if max_round_age_seconds < 1 or limit < 1:
        raise ValueError("Freshness age and limit must be positive")
    cutoff = utc_iso(utc_now() - timedelta(seconds=max_round_age_seconds))
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, payload_json, actual_status, wikidata_entity_id,
                   portrait_url, validated_at
            FROM round_inventory
            WHERE state = 'available'
              AND (validated_at IS NULL OR datetime(validated_at) <= datetime(?))
            ORDER BY validated_at, id
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
    result = {"examined": len(rows), "refreshed": 0, "retired": 0, "errors": 0}
    for row in rows:
        attempted_at = utc_iso()
        try:
            payload = validate_round_payload(json.loads(row["payload_json"]))
            facts = revalidate(payload)
            new_status = str(facts["actual_status"])
            new_entity_id = str(facts.get("wikidata_entity_id") or "")
            new_portrait_url = str(facts.get("portrait_url") or "")
            identity_changed = bool(
                row["wikidata_entity_id"]
                and new_entity_id
                and row["wikidata_entity_id"] != new_entity_id
            )
            portrait_changed = bool(
                row["portrait_url"]
                and new_portrait_url
                and row["portrait_url"] != new_portrait_url
            )
            status_changed = new_status != row["actual_status"]
            if identity_changed or portrait_changed or status_changed:
                reasons = []
                if identity_changed:
                    reasons.append("identity changed")
                if portrait_changed:
                    reasons.append("portrait changed")
                if status_changed:
                    reasons.append("vital status changed")
                with connect(db_path) as connection:
                    updated = connection.execute(
                        """
                        UPDATE round_inventory
                        SET state = 'retired', validation_attempted_at = ?,
                            validation_error = ?, updated_at = ?
                        WHERE id = ? AND state = 'available'
                        """,
                        (attempted_at, ", ".join(reasons), attempted_at, row["id"]),
                    ).rowcount
                result["retired"] += updated
                continue
            with connect(db_path) as connection:
                updated = connection.execute(
                    """
                    UPDATE round_inventory
                    SET validated_at = ?, validation_attempted_at = ?,
                        validation_error = NULL, wikidata_entity_id = ?,
                        wikipedia_title = ?, birth_date = ?, portrait_url = ?,
                        updated_at = ?
                    WHERE id = ? AND state = 'available'
                    """,
                    (
                        attempted_at,
                        attempted_at,
                        new_entity_id or row["wikidata_entity_id"],
                        facts.get("wikipedia_title"),
                        facts.get("birth_date"),
                        new_portrait_url or row["portrait_url"],
                        attempted_at,
                        row["id"],
                    ),
                ).rowcount
            result["refreshed"] += updated
        except Exception as exc:
            with connect(db_path) as connection:
                updated = connection.execute(
                    """
                    UPDATE round_inventory
                    SET validation_attempted_at = ?, validation_error = ?, updated_at = ?
                    WHERE id = ? AND state = 'available'
                    """,
                    (attempted_at, str(exc)[:2000], attempted_at, row["id"]),
                ).rowcount
            result["errors"] += updated
    return result


def retire_stale_available_packs(
    db_path: Path | str,
    *,
    max_round_age_seconds: int,
) -> int:
    if max_round_age_seconds < 1:
        raise ValueError("max_round_age_seconds must be positive")
    cutoff = utc_iso(utc_now() - timedelta(seconds=max_round_age_seconds))
    connection = connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute(
            """
            SELECT DISTINCT p.id FROM game_packs AS p
            JOIN game_pack_rounds AS pr ON pr.pack_id = p.id
            JOIN round_inventory AS r ON r.id = pr.round_id
            WHERE p.state = 'available'
              AND (r.validated_at IS NULL OR datetime(r.validated_at) <= datetime(?))
            """,
            (cutoff,),
        ).fetchall()
        for row in rows:
            now = utc_iso()
            connection.execute(
                "UPDATE game_packs SET state = 'retired', updated_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            connection.execute(
                """
                UPDATE round_inventory
                SET state = 'retired', validation_error = 'pack expired', updated_at = ?
                WHERE state = 'packed'
                  AND id IN (SELECT round_id FROM game_pack_rounds WHERE pack_id = ?)
                """,
                (now, row["id"]),
            )
        connection.commit()
        return len(rows)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _reclaim_expired_reservations(
    connection: sqlite3.Connection,
    now: str,
) -> int:
    expired_rows = connection.execute(
        """
        SELECT id, reserved_by FROM game_packs
        WHERE state = 'reserved' AND reservation_expires_at IS NOT NULL
          AND datetime(reservation_expires_at) <= datetime(?)
        """,
        (now,),
    ).fetchall()
    for row in expired_rows:
        connection.execute(
            """
            UPDATE round_inventory
            SET state = 'packed', reserved_by = NULL, updated_at = ?
            WHERE state = 'reserved' AND reserved_by = ?
              AND id IN (SELECT round_id FROM game_pack_rounds WHERE pack_id = ?)
            """,
            (now, row["reserved_by"], row["id"]),
        )
    if expired_rows:
        pack_ids = [int(row["id"]) for row in expired_rows]
        placeholders = ",".join("?" for _ in pack_ids)
        connection.execute(
            f"UPDATE game_packs SET state = 'available', reserved_by = NULL, "
            f"reserved_at = NULL, reservation_expires_at = NULL, updated_at = ? "
            f"WHERE id IN ({placeholders})",
            (now, *pack_ids),
        )
    return len(expired_rows)


def reclaim_expired_reservations(db_path: Path | str) -> int:
    connection = connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        reclaimed = _reclaim_expired_reservations(connection, utc_iso())
        connection.commit()
        return reclaimed
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def reserve_classic_pack(
    db_path: Path | str,
    category: str,
    session_id: str,
    *,
    round_count: int = 10,
    reservation_ttl_seconds: int = 7200,
    max_round_age_seconds: int | None = None,
) -> dict | None:
    """Reserve one ready pack with a single SQLite write transaction."""
    connection = connect(db_path)
    try:
        initialize_schema(connection)
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        _reclaim_expired_reservations(connection, utc_iso())
        freshness_clause = ""
        parameters: list[object] = [normalize_category_key(category), round_count]
        if max_round_age_seconds is not None:
            if max_round_age_seconds < 1:
                raise ValueError("max_round_age_seconds must be positive")
            freshness_clause = """
              AND NOT EXISTS (
                  SELECT 1 FROM game_pack_rounds AS pr
                  JOIN round_inventory AS r ON r.id = pr.round_id
                  WHERE pr.pack_id = game_packs.id
                    AND (r.validated_at IS NULL OR datetime(r.validated_at) <= datetime(?))
              )
            """
            parameters.append(
                utc_iso(utc_now() - timedelta(seconds=max_round_age_seconds))
            )
        pack = connection.execute(
            f"""
            SELECT id FROM game_packs
            WHERE category_key = ? AND mode = 'classic' AND state = 'available'
              AND round_count = ?
              {freshness_clause}
            ORDER BY created_at, id
            LIMIT 1
            """,
            parameters,
        ).fetchone()
        if pack is None:
            connection.rollback()
            return None
        pack_id = int(pack["id"])
        now_value = utc_now()
        now = utc_iso(now_value)
        expires_at = utc_iso(now_value + timedelta(seconds=reservation_ttl_seconds))
        updated = connection.execute(
            """
            UPDATE game_packs
            SET state = 'reserved', reserved_by = ?, reserved_at = ?,
                reservation_expires_at = ?, updated_at = ?
            WHERE id = ? AND state = 'available'
            """,
            (session_id, now, expires_at, now, pack_id),
        ).rowcount
        if updated != 1:
            connection.rollback()
            return None
        reserved_rounds = connection.execute(
            """
            UPDATE round_inventory
            SET state = 'reserved', reserved_by = ?, updated_at = ?
            WHERE state = 'packed' AND id IN (
                SELECT round_id FROM game_pack_rounds WHERE pack_id = ?
            )
            """,
            (session_id, now, pack_id),
        ).rowcount
        if reserved_rounds != round_count:
            raise RuntimeError(f"Prepared pack {pack_id} does not have {round_count} reservable rounds")
        rows = connection.execute(
            """
            SELECT r.payload_json, r.payload_sha256
            FROM game_pack_rounds AS pr
            JOIN round_inventory AS r ON r.id = pr.round_id
            WHERE pr.pack_id = ?
            ORDER BY pr.position
            """,
            (pack_id,),
        ).fetchall()
        if len(rows) != round_count:
            raise RuntimeError(f"Prepared pack {pack_id} does not contain {round_count} rounds")
        rounds = []
        for row in rows:
            payload_json = str(row["payload_json"])
            payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            if payload_hash != row["payload_sha256"]:
                raise RuntimeError(f"Prepared pack {pack_id} contains a corrupted round payload")
            rounds.append(validate_round_payload(json.loads(payload_json)))
        connection.commit()
        return {
            "pack_id": pack_id,
            "session_id": session_id,
            "rounds": rounds,
            "reservation_expires_at": expires_at,
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def release_pack(db_path: Path | str, session_id: str) -> bool:
    connection = connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        pack = connection.execute(
            "SELECT id FROM game_packs WHERE reserved_by = ? AND state = 'reserved'",
            (session_id,),
        ).fetchone()
        if pack is None:
            connection.rollback()
            return False
        pack_id = int(pack["id"])
        now = utc_iso()
        connection.execute(
            """
            UPDATE round_inventory
            SET state = 'packed', reserved_by = NULL, updated_at = ?
            WHERE reserved_by = ? AND state = 'reserved'
              AND id IN (SELECT round_id FROM game_pack_rounds WHERE pack_id = ?)
            """,
            (now, session_id, pack_id),
        )
        connection.execute(
            """
            UPDATE game_packs
            SET state = 'available', reserved_by = NULL, reserved_at = NULL,
                reservation_expires_at = NULL, updated_at = ?
            WHERE id = ? AND state = 'reserved'
            """,
            (now, pack_id),
        )
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def consume_pack(db_path: Path | str, session_id: str) -> bool:
    connection = connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        pack = connection.execute(
            "SELECT id FROM game_packs WHERE reserved_by = ? AND state = 'reserved'",
            (session_id,),
        ).fetchone()
        if pack is None:
            connection.rollback()
            return False
        pack_id = int(pack["id"])
        now = utc_iso()
        connection.execute(
            """
            UPDATE round_inventory SET state = 'consumed', updated_at = ?
            WHERE reserved_by = ? AND state = 'reserved'
              AND id IN (SELECT round_id FROM game_pack_rounds WHERE pack_id = ?)
            """,
            (now, session_id, pack_id),
        )
        connection.execute(
            """
            UPDATE game_packs
            SET state = 'consumed', reservation_expires_at = NULL, updated_at = ?
            WHERE id = ? AND state = 'reserved'
            """,
            (now, pack_id),
        )
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def inventory_health(
    db_path: Path | str,
    *,
    max_round_age_seconds: int = 86400,
) -> dict:
    cutoff = utc_iso(utc_now() - timedelta(seconds=max_round_age_seconds))
    with connect(db_path) as connection:
        round_rows = connection.execute(
            "SELECT state, COUNT(*) AS count FROM round_inventory GROUP BY state"
        ).fetchall()
        pack_rows = connection.execute(
            "SELECT state, COUNT(*) AS count FROM game_packs GROUP BY state"
        ).fetchall()
        job_rows = connection.execute(
            "SELECT state, COUNT(*) AS count FROM inventory_refill_jobs GROUP BY state"
        ).fetchall()
        backlog = connection.execute(
            """
            SELECT COALESCE(SUM(
                requested_packs - (
                    SELECT COUNT(*) FROM game_packs AS p
                    WHERE p.source_job_id = inventory_refill_jobs.id
                      AND p.state != 'retired'
                )
            ), 0)
            FROM inventory_refill_jobs
            WHERE state IN ('queued', 'leased')
            """
        ).fetchone()[0]
        stale_available_rounds = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM round_inventory
                WHERE state = 'available'
                  AND (validated_at IS NULL OR datetime(validated_at) <= datetime(?))
                """,
                (cutoff,),
            ).fetchone()[0]
        )
        available_pack_rows = connection.execute(
            """
            SELECT category, COUNT(*) AS count FROM game_packs
            WHERE state = 'available' GROUP BY category ORDER BY category
            """
        ).fetchall()
    rounds = {state: 0 for state in ROUND_STATES}
    packs = {state: 0 for state in PACK_STATES}
    jobs = {state: 0 for state in JOB_STATES}
    rounds.update({str(row["state"]): int(row["count"]) for row in round_rows})
    packs.update({str(row["state"]): int(row["count"]) for row in pack_rows})
    jobs.update({str(row["state"]): int(row["count"]) for row in job_rows})
    return {
        "schema_version": SCHEMA_VERSION,
        "rounds": rounds,
        "packs": packs,
        "jobs": jobs,
        "backlog_packs": max(0, int(backlog)),
        "stale_available_rounds": stale_available_rounds,
        "available_packs_by_category": {
            str(row["category"]): int(row["count"]) for row in available_pack_rows
        },
    }
