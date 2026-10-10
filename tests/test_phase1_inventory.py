import asyncio
import concurrent.futures
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient

import inventory
from inventory_worker import process_one_job
import main
from refill_inventory import make_main_round_revalidator, refill_inventory


def round_payload(number: int, *, category: str = "Test") -> dict:
    person_name = f"{category} Person {number}"
    portrait_url = f"https://upload.wikimedia.org/portrait-{number}.jpg"
    return {
        "person_name": person_name,
        "actual_status": "alive" if number % 2 else "dead",
        "guessing_ui_html": (
            f'<div><img src="{portrait_url}"><button onclick="submitGuess(\'alive\')">'
            "Alive</button><button onclick=\"submitGuess('dead')\">Dead</button></div>"
        ),
        "reveal_ui_html": '<div><button onclick="loadNextRound()">Next</button></div>',
        "wikidata_entity_id": f"Q{900000 + number}",
        "wikipedia_title": person_name,
        "birth_date": "+1980-01-01T00:00:00Z",
        "portrait_url": portrait_url,
    }


def app_session(*, round_state: str = "revealed", round_number: int = 1) -> dict:
    now = time.monotonic()
    return {
        "score": 0,
        "round_number": round_number,
        "history": [],
        "used_names": [],
        "active_round": None,
        "queued_rounds": [],
        "prefetch_task": None,
        "prefetch_error": None,
        "inventory_pack_id": None,
        "inventory_rounds": [],
        "created_at": now,
        "last_activity": now,
        "round_lock": asyncio.Lock(),
        "state_lock": asyncio.Lock(),
        "round_state": round_state,
        "leaderboard_submission_state": "available",
        "client_key": "testclient",
        "mode": "classic",
        "category": "All Celebrities",
    }


class InventoryPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "inventory.db"
        with inventory.connect(self.db_path) as connection:
            inventory.initialize_schema(connection)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_rounds(self, count: int, category: str = "All Celebrities") -> None:
        for number in range(1, count + 1):
            inventory.add_round(
                self.db_path,
                category,
                round_payload(number, category=category),
                source_model="test-model",
            )

    def test_schema_migration_is_additive_and_versioned(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "CREATE TABLE legacy_data(id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute("INSERT INTO legacy_data(value) VALUES ('preserved')")
            inventory.initialize_schema(connection)
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            migration = connection.execute(
                "SELECT version FROM app_schema_migrations WHERE component = 'round_inventory'"
            ).fetchone()
            round_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(round_inventory)")
            }
            pack_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(game_packs)")
            }
            legacy_value = connection.execute("SELECT value FROM legacy_data").fetchone()[0]

        self.assertIn("round_inventory", tables)
        self.assertIn("game_packs", tables)
        self.assertIn("game_pack_rounds", tables)
        self.assertIn("validated_at", round_columns)
        self.assertIn("validation_error", round_columns)
        self.assertIn("source_job_id", pack_columns)
        self.assertEqual(migration[0], inventory.SCHEMA_VERSION)
        self.assertEqual(legacy_value, "preserved")

    def test_schema_migrates_existing_phase1_inventory_without_losing_rounds(self) -> None:
        legacy_path = Path(self.temp_dir.name) / "phase1-v1.db"
        created_at = "2026-07-18T00:00:00Z"
        with sqlite3.connect(legacy_path) as connection:
            connection.executescript(
                """
                CREATE TABLE round_inventory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category_key TEXT NOT NULL, category TEXT NOT NULL,
                    person_key TEXT NOT NULL, person_name TEXT NOT NULL,
                    actual_status TEXT NOT NULL, payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL, payload_version INTEGER NOT NULL DEFAULT 1,
                    source_model TEXT, wikidata_entity_id TEXT, wikipedia_title TEXT,
                    birth_date TEXT, portrait_url TEXT, state TEXT NOT NULL,
                    reserved_by TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    expires_at TEXT, UNIQUE(category_key, person_key)
                );
                CREATE TABLE game_packs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category_key TEXT NOT NULL, category TEXT NOT NULL,
                    mode TEXT NOT NULL, round_count INTEGER NOT NULL,
                    state TEXT NOT NULL, reserved_by TEXT, reserved_at TEXT,
                    reservation_expires_at TEXT, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE game_pack_rounds (
                    pack_id INTEGER NOT NULL, round_id INTEGER NOT NULL UNIQUE,
                    position INTEGER NOT NULL, PRIMARY KEY(pack_id, position)
                );
                """
            )
            payload = round_payload(1, category="Legacy")
            payload_json = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            connection.execute(
                """
                INSERT INTO round_inventory (
                    category_key, category, person_key, person_name, actual_status,
                    payload_json, payload_sha256, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'available', ?, ?)
                """,
                (
                    "legacy",
                    "Legacy",
                    "legacy person 1",
                    "Legacy Person 1",
                    "alive",
                    payload_json,
                    "legacy-hash",
                    created_at,
                    created_at,
                ),
            )
            inventory.initialize_schema(connection)
            round_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(round_inventory)")
            }
            pack_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(game_packs)")
            }
            migrated = connection.execute(
                "SELECT person_name, validated_at FROM round_inventory"
            ).fetchone()

        self.assertIn("validated_at", round_columns)
        self.assertIn("validation_error", round_columns)
        self.assertIn("source_job_id", pack_columns)
        self.assertEqual(migrated, ("Legacy Person 1", created_at))

    def test_concurrent_reservations_never_return_the_same_pack(self) -> None:
        self.add_rounds(20)
        expected_pack_ids = {
            inventory.create_classic_pack(self.db_path, "All Celebrities"),
            inventory.create_classic_pack(self.db_path, "All Celebrities"),
        }

        def reserve(index: int):
            return inventory.reserve_classic_pack(
                self.db_path,
                "All Celebrities",
                f"session-{index}",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            reservations = list(executor.map(reserve, range(4)))

        reserved = [reservation for reservation in reservations if reservation]
        self.assertEqual(len(reserved), 2)
        self.assertEqual({item["pack_id"] for item in reserved}, expected_pack_ids)
        self.assertEqual(len({item["session_id"] for item in reserved}), 2)
        self.assertTrue(all(len(item["rounds"]) == 10 for item in reserved))

    def test_depleted_inventory_is_explicit_and_released_pack_can_be_reused(self) -> None:
        self.add_rounds(10)
        pack_id = inventory.create_classic_pack(self.db_path, "All Celebrities")
        first = inventory.reserve_classic_pack(
            self.db_path,
            "All Celebrities",
            "first-session",
        )
        depleted = inventory.reserve_classic_pack(
            self.db_path,
            "All Celebrities",
            "second-session",
        )

        self.assertEqual(first["pack_id"], pack_id)
        self.assertIsNone(depleted)
        self.assertTrue(inventory.release_pack(self.db_path, "first-session"))
        second = inventory.reserve_classic_pack(
            self.db_path,
            "All Celebrities",
            "second-session",
        )
        self.assertEqual(second["pack_id"], pack_id)

    def test_expired_reservation_is_reclaimed_after_restart(self) -> None:
        self.add_rounds(10)
        pack_id = inventory.create_classic_pack(self.db_path, "All Celebrities")
        first = inventory.reserve_classic_pack(
            self.db_path,
            "All Celebrities",
            "abandoned-session",
            reservation_ttl_seconds=-1,
        )

        second = inventory.reserve_classic_pack(
            self.db_path,
            "All Celebrities",
            "replacement-session",
        )

        self.assertEqual(first["pack_id"], pack_id)
        self.assertEqual(second["pack_id"], pack_id)
        self.assertEqual(second["session_id"], "replacement-session")

    def test_corrupted_pack_payload_cannot_be_reserved(self) -> None:
        self.add_rounds(10)
        inventory.create_classic_pack(self.db_path, "All Celebrities")
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE round_inventory SET payload_json = '{}' WHERE id = 1"
            )

        with self.assertRaisesRegex(RuntimeError, "corrupted round payload"):
            inventory.reserve_classic_pack(
                self.db_path,
                "All Celebrities",
                "corruption-test",
            )

        health = inventory.inventory_health(self.db_path)
        self.assertEqual(health["packs"]["available"], 1)
        self.assertEqual(health["packs"]["reserved"], 0)

    def test_refill_command_builds_only_complete_packs(self) -> None:
        generated = iter(range(1, 6))

        def fake_generate(_forbidden: list[str], category: str):
            number = next(generated)
            return "offline-test-model", round_payload(number, category=category)

        result = refill_inventory(
            self.db_path,
            "Test Category",
            rounds_to_generate=5,
            packs_to_create=2,
            rounds_per_pack=2,
            generate_round=fake_generate,
        )

        self.assertEqual(result["generated_rounds"], 5)
        self.assertEqual(result["created_packs"], 2)
        self.assertEqual(result["inventory"]["packs"]["available"], 2)
        self.assertEqual(result["inventory"]["rounds"]["available"], 1)
        self.assertEqual(result["inventory"]["rounds"]["packed"], 4)

    def test_balanced_pack_requires_and_selects_exact_status_counts(self) -> None:
        for number in range(1, 9):
            payload = round_payload(number, category="Balanced")
            payload["actual_status"] = "alive"
            inventory.add_round(self.db_path, "Balanced", payload)
        for number in range(9, 13):
            payload = round_payload(number, category="Balanced")
            payload["actual_status"] = "dead"
            inventory.add_round(self.db_path, "Balanced", payload)

        unavailable = inventory.create_classic_pack(
            self.db_path,
            "Balanced",
            round_count=10,
            alive_per_pack=5,
        )
        final_dead = round_payload(13, category="Balanced")
        final_dead["actual_status"] = "dead"
        inventory.add_round(self.db_path, "Balanced", final_dead)
        pack_id = inventory.create_classic_pack(
            self.db_path,
            "Balanced",
            round_count=10,
            alive_per_pack=5,
        )

        self.assertIsNone(unavailable)
        self.assertIsNotNone(pack_id)
        with inventory.connect(self.db_path) as connection:
            statuses = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT r.actual_status FROM game_pack_rounds AS pr
                    JOIN round_inventory AS r ON r.id = pr.round_id
                    WHERE pr.pack_id = ? ORDER BY pr.position
                    """,
                    (pack_id,),
                )
            ]
        self.assertEqual(statuses.count("alive"), 5)
        self.assertEqual(statuses.count("dead"), 5)
        self.assertEqual(statuses, ["alive", "dead"] * 5)

    def test_minimum_stock_planner_enqueues_only_the_pack_deficit(self) -> None:
        first = inventory.ensure_minimum_pack_stock(
            self.db_path,
            "All Celebrities",
            minimum_ready_packs=3,
            rounds_per_pack=10,
            alive_per_pack=5,
        )
        second = inventory.ensure_minimum_pack_stock(
            self.db_path,
            "All Celebrities",
            minimum_ready_packs=3,
            rounds_per_pack=10,
            alive_per_pack=5,
        )

        self.assertEqual(first["ready_packs"], 0)
        self.assertEqual(first["deficit_packs"], 3)
        self.assertEqual(first["job"]["requested_packs"], 3)
        self.assertTrue(first["job"]["created"])
        self.assertFalse(second["job"]["created"])
        self.assertEqual(second["job"]["id"], first["job"]["id"])

    def test_minimum_stock_planner_does_not_enqueue_when_target_is_met(self) -> None:
        self.add_rounds(10)
        inventory.create_classic_pack(
            self.db_path,
            "All Celebrities",
            alive_per_pack=5,
        )

        result = inventory.ensure_minimum_pack_stock(
            self.db_path,
            "All Celebrities",
            minimum_ready_packs=1,
            rounds_per_pack=10,
            alive_per_pack=5,
        )

        self.assertEqual(result["ready_packs"], 1)
        self.assertEqual(result["deficit_packs"], 0)
        self.assertIsNone(result["job"])

    def test_minimum_stock_planner_honors_failed_job_cooldown(self) -> None:
        job = inventory.enqueue_refill_job(
            self.db_path,
            "All Celebrities",
            requested_packs=1,
            max_attempts=1,
        )
        inventory.claim_refill_job(self.db_path, "failed-worker")
        inventory.fail_refill_job(
            self.db_path,
            job["id"],
            "failed-worker",
            "provider unavailable",
        )

        blocked = inventory.ensure_minimum_pack_stock(
            self.db_path,
            "All Celebrities",
            minimum_ready_packs=1,
            failed_job_cooldown_seconds=3600,
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE inventory_refill_jobs SET updated_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", job["id"]),
            )
        resumed = inventory.ensure_minimum_pack_stock(
            self.db_path,
            "All Celebrities",
            minimum_ready_packs=1,
            failed_job_cooldown_seconds=3600,
        )

        self.assertIsNone(blocked["job"])
        self.assertEqual(blocked["blocked_by_failed_job"]["id"], job["id"])
        self.assertTrue(resumed["job"]["created"])
        self.assertIsNone(resumed["blocked_by_failed_job"])

    def test_stale_rounds_are_refreshed_retired_or_quarantined_on_error(self) -> None:
        round_ids = []
        for number in range(1, 4):
            round_ids.append(
                inventory.add_round(
                    self.db_path,
                    "Freshness",
                    round_payload(number, category="Freshness"),
                )
            )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE round_inventory SET validated_at = '2000-01-01T00:00:00Z'"
            )

        def revalidate(payload: dict) -> dict:
            if payload["person_name"].endswith("3"):
                raise RuntimeError("verification service unavailable")
            status = payload["actual_status"]
            if payload["person_name"].endswith("2"):
                status = "alive" if status == "dead" else "dead"
            return {
                "actual_status": status,
                "wikidata_entity_id": payload["wikidata_entity_id"],
                "wikipedia_title": payload["wikipedia_title"],
                "birth_date": payload["birth_date"],
                "portrait_url": payload["portrait_url"],
            }

        result = inventory.revalidate_stale_available_rounds(
            self.db_path,
            revalidate,
            max_round_age_seconds=3600,
        )

        self.assertEqual(
            result,
            {"examined": 3, "refreshed": 1, "retired": 1, "errors": 1},
        )
        with inventory.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, state, validated_at, validation_error
                FROM round_inventory ORDER BY id
                """
            ).fetchall()
        self.assertEqual(rows[0]["state"], "available")
        self.assertNotEqual(rows[0]["validated_at"], "2000-01-01T00:00:00Z")
        self.assertIsNone(rows[0]["validation_error"])
        self.assertEqual(rows[1]["state"], "retired")
        self.assertIn("vital status changed", rows[1]["validation_error"])
        self.assertEqual(rows[2]["state"], "available")
        self.assertEqual(rows[2]["validated_at"], "2000-01-01T00:00:00Z")
        self.assertIn("verification service unavailable", rows[2]["validation_error"])
        pack_id = inventory.create_classic_pack(
            self.db_path,
            "Freshness",
            round_count=1,
            alive_per_pack=1,
            max_round_age_seconds=3600,
        )
        self.assertIsNotNone(pack_id)

    def test_stale_pack_cannot_be_reserved_and_is_retired_by_maintenance(self) -> None:
        self.add_rounds(10)
        pack_id = inventory.create_classic_pack(
            self.db_path,
            "All Celebrities",
            alive_per_pack=5,
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE round_inventory SET validated_at = '2000-01-01T00:00:00Z'
                WHERE id = (SELECT round_id FROM game_pack_rounds
                            WHERE pack_id = ? ORDER BY position LIMIT 1)
                """,
                (pack_id,),
            )

        reservation = inventory.reserve_classic_pack(
            self.db_path,
            "All Celebrities",
            "freshness-session",
            max_round_age_seconds=3600,
        )
        retired = inventory.retire_stale_available_packs(
            self.db_path,
            max_round_age_seconds=3600,
        )
        health = inventory.inventory_health(self.db_path)

        self.assertIsNone(reservation)
        self.assertEqual(retired, 1)
        self.assertEqual(health["packs"]["retired"], 1)
        self.assertEqual(health["rounds"]["retired"], 10)

    def test_corrupted_stale_round_is_quarantined_without_stopping_maintenance(self) -> None:
        round_id = inventory.add_round(
            self.db_path,
            "Freshness",
            round_payload(1, category="Freshness"),
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE round_inventory
                SET payload_json = '{}', validated_at = '2000-01-01T00:00:00Z'
                WHERE id = ?
                """,
                (round_id,),
            )

        result = inventory.revalidate_stale_available_rounds(
            self.db_path,
            lambda _payload: self.fail("corrupted payload must not reach revalidator"),
            max_round_age_seconds=3600,
        )

        self.assertEqual(result["errors"], 1)
        with inventory.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT state, validation_error FROM round_inventory WHERE id = ?",
                (round_id,),
            ).fetchone()
        self.assertEqual(row["state"], "available")
        self.assertIn("missing required fields", row["validation_error"])


class RefillJobLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "jobs.db"
        with inventory.connect(self.db_path) as connection:
            inventory.initialize_schema(connection)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def enqueue(self, *, max_attempts: int = 3) -> dict:
        return inventory.enqueue_refill_job(
            self.db_path,
            "All Celebrities",
            requested_packs=2,
            rounds_per_pack=10,
            alive_per_pack=5,
            max_attempts=max_attempts,
        )

    def test_enqueue_is_idempotent_while_category_job_is_active(self) -> None:
        first = self.enqueue()
        second = self.enqueue()

        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(second["id"], first["id"])
        with sqlite3.connect(self.db_path) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM inventory_refill_jobs"
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_concurrent_workers_cannot_lease_the_same_job(self) -> None:
        expected = self.enqueue()

        def claim(index: int):
            return inventory.claim_refill_job(self.db_path, f"worker-{index}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            claims = list(executor.map(claim, range(4)))

        leased = [claim for claim in claims if claim is not None]
        self.assertEqual(len(leased), 1)
        self.assertEqual(leased[0]["id"], expected["id"])
        self.assertEqual(leased[0]["attempt_count"], 1)

    def test_expired_lease_is_recovered_after_worker_restart(self) -> None:
        job = self.enqueue()
        first = inventory.claim_refill_job(self.db_path, "stopped-worker")
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE inventory_refill_jobs SET lease_expires_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", job["id"]),
            )

        recovered = inventory.claim_refill_job(self.db_path, "replacement-worker")

        self.assertEqual(first["id"], job["id"])
        self.assertEqual(recovered["id"], job["id"])
        self.assertEqual(recovered["lease_owner"], "replacement-worker")
        self.assertEqual(recovered["attempt_count"], 2)

    def test_lease_renewal_completion_and_retry_require_the_owner(self) -> None:
        job = self.enqueue(max_attempts=2)
        leased = inventory.claim_refill_job(self.db_path, "worker-a")

        self.assertFalse(
            inventory.renew_refill_job_lease(
                self.db_path,
                job["id"],
                "worker-b",
            )
        )
        self.assertTrue(
            inventory.renew_refill_job_lease(
                self.db_path,
                job["id"],
                "worker-a",
            )
        )
        self.assertIsNone(
            inventory.fail_refill_job(
                self.db_path,
                job["id"],
                "worker-b",
                "not the owner",
            )
        )
        self.assertEqual(
            inventory.fail_refill_job(
                self.db_path,
                job["id"],
                "worker-a",
                "temporary failure",
                retry_delay_seconds=0,
            ),
            "queued",
        )
        retry = inventory.claim_refill_job(self.db_path, "worker-c")
        self.assertEqual(retry["attempt_count"], 2)
        self.assertTrue(
            inventory.complete_refill_job(
                self.db_path,
                job["id"],
                "worker-c",
            )
        )
        self.assertFalse(
            inventory.complete_refill_job(
                self.db_path,
                job["id"],
                "worker-c",
            )
        )
        self.assertEqual(leased["id"], job["id"])

    def test_health_reports_job_backlog_category_stock_and_stale_rounds(self) -> None:
        job = self.enqueue()
        inventory.claim_refill_job(self.db_path, "health-worker")
        for number in range(1, 11):
            inventory.add_round(
                self.db_path,
                "All Celebrities",
                round_payload(number, category="All Celebrities"),
            )
        inventory.create_classic_pack(
            self.db_path,
            "All Celebrities",
            alive_per_pack=5,
            source_job_id=job["id"],
        )
        stale_id = inventory.add_round(
            self.db_path,
            "All Celebrities",
            round_payload(11, category="All Celebrities"),
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE round_inventory SET validated_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", stale_id),
            )

        health = inventory.inventory_health(
            self.db_path,
            max_round_age_seconds=3600,
        )

        self.assertEqual(health["jobs"]["leased"], 1)
        self.assertEqual(health["backlog_packs"], 1)
        self.assertEqual(health["stale_available_rounds"], 1)
        self.assertEqual(health["available_packs_by_category"]["All Celebrities"], 1)


class InventoryWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "worker.db"
        with inventory.connect(self.db_path) as connection:
            inventory.initialize_schema(connection)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def enqueue(self, *, packs: int = 1, max_attempts: int = 3) -> dict:
        return inventory.enqueue_refill_job(
            self.db_path,
            "Worker Category",
            requested_packs=packs,
            rounds_per_pack=4,
            alive_per_pack=2,
            max_attempts=max_attempts,
        )

    def test_worker_generates_balanced_pack_and_completes_job(self) -> None:
        job = self.enqueue()
        numbers = iter(range(1, 10))

        def generate(_forbidden: list[str], category: str):
            number = next(numbers)
            return "worker-test-model", round_payload(number, category=category)

        result = process_one_job(
            self.db_path,
            "worker-1",
            generate,
            lease_ttl_seconds=60,
            max_generated_rounds_per_pack=8,
        )

        self.assertEqual(result["job_id"], job["id"])
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["generated_rounds"], 4)
        self.assertEqual(result["total_job_packs"], 1)
        with inventory.connect(self.db_path) as connection:
            job_state = connection.execute(
                "SELECT state FROM inventory_refill_jobs WHERE id = ?",
                (job["id"],),
            ).fetchone()[0]
            source_job_id = connection.execute(
                "SELECT source_job_id FROM game_packs"
            ).fetchone()[0]
        self.assertEqual(job_state, "completed")
        self.assertEqual(source_job_id, job["id"])

    def test_recovered_worker_counts_pack_created_before_crash(self) -> None:
        job = self.enqueue(packs=2)
        inventory.claim_refill_job(self.db_path, "crashed-worker", lease_ttl_seconds=60)
        for number in range(1, 5):
            inventory.add_round(
                self.db_path,
                "Worker Category",
                round_payload(number, category="Worker Category"),
            )
        first_pack = inventory.create_classic_pack(
            self.db_path,
            "Worker Category",
            round_count=4,
            alive_per_pack=2,
            source_job_id=job["id"],
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE inventory_refill_jobs SET lease_expires_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", job["id"]),
            )
        generated_numbers = iter(range(5, 20))
        calls = []

        def generate(_forbidden: list[str], category: str):
            number = next(generated_numbers)
            calls.append(number)
            return "worker-test-model", round_payload(number, category=category)

        result = process_one_job(
            self.db_path,
            "replacement-worker",
            generate,
            lease_ttl_seconds=60,
            max_generated_rounds_per_pack=8,
        )

        self.assertIsNotNone(first_pack)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["total_job_packs"], 2)
        self.assertEqual(len(calls), 4)

    def test_worker_failure_retries_then_exhausts_job(self) -> None:
        job = self.enqueue(max_attempts=2)

        def fail_generation(_forbidden: list[str], _category: str):
            raise RuntimeError("generator unavailable")

        first = process_one_job(
            self.db_path,
            "worker-1",
            fail_generation,
            retry_delay_seconds=0,
        )
        second = process_one_job(
            self.db_path,
            "worker-2",
            fail_generation,
            retry_delay_seconds=0,
        )

        self.assertEqual(first["state"], "queued")
        self.assertEqual(second["state"], "failed")
        with inventory.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT state, attempt_count, last_error FROM inventory_refill_jobs WHERE id = ?",
                (job["id"],),
            ).fetchone()
        self.assertEqual(row["state"], "failed")
        self.assertEqual(row["attempt_count"], 2)
        self.assertIn("generator unavailable", row["last_error"])

    def test_revalidator_bypasses_status_identity_and_portrait_caches(self) -> None:
        payload = round_payload(1, category="Revalidation")
        facts = main.VerifiedPersonFacts(
            actual_status=payload["actual_status"],
            birth_date=payload["birth_date"],
            wikipedia_title=payload["wikipedia_title"],
            wikidata_entity_id=payload["wikidata_entity_id"],
        )
        revalidate = make_main_round_revalidator(main)

        with (
            patch.object(main, "verify_person_facts", return_value=facts) as verify,
            patch.object(
                main,
                "get_round_portrait_url",
                return_value=payload["portrait_url"],
            ),
            patch.object(main, "validate_image_url_reachable") as validate_portrait,
        ):
            result = revalidate(payload)

        verify.assert_called_once_with(payload["person_name"], use_cache=False)
        validate_portrait.assert_called_once_with(
            payload["portrait_url"],
            payload["person_name"],
            "Inventory revalidation",
            use_cache=False,
        )
        self.assertEqual(result["wikidata_entity_id"], payload["wikidata_entity_id"])

    def test_fresh_fact_verification_bypasses_all_identity_caches(self) -> None:
        person_name = "Fresh Verification Person"
        summary = {"wikibase_item": "Q999001"}
        with (
            patch.object(main.status_verification_cache, "get") as status_cache,
            patch.object(
                main,
                "resolve_wikipedia_person_summary",
                return_value=(summary, person_name),
            ) as resolve,
            patch.object(main, "get_wikidata_entity", return_value={}) as get_entity,
            patch.object(main, "wikidata_claim_has_entity_value", return_value=True),
            patch.object(main, "validate_candidate_birth_date", return_value="+1980-01-01T00:00:00Z"),
            patch.object(main, "wikidata_claim_has_value", return_value=False),
            patch.object(main, "append_gemini_audit_log"),
        ):
            facts = main.verify_person_facts(person_name, use_cache=False)

        status_cache.assert_not_called()
        resolve.assert_called_once_with(person_name, use_cache=False)
        get_entity.assert_called_once_with("Q999001", use_cache=False)
        self.assertEqual(facts.actual_status, "alive")


class InventoryServingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "app.db"
        self.original_db_path = main.DB_PATH
        main.DB_PATH = self.db_path
        main.init_db()
        main.sessions.clear()
        main.start_session_rate_limiter.clear()
        main.timing_telemetry.clear()
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        main.sessions.clear()
        main.start_session_rate_limiter.clear()
        self.client_context.__exit__(None, None, None)
        main.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def stock_pack(self) -> int:
        for number in range(1, main.ROUNDS_PER_GAME + 1):
            inventory.add_round(
                self.db_path,
                "All Celebrities",
                round_payload(number, category="All Celebrities"),
            )
        return inventory.create_classic_pack(
            self.db_path,
            "All Celebrities",
            round_count=main.ROUNDS_PER_GAME,
        )

    def test_enabled_classic_session_uses_reserved_pack_without_generation(self) -> None:
        expected_pack_id = self.stock_pack()
        with (
            patch.object(main, "ROUND_INVENTORY_ENABLED", True),
            patch.object(main, "generate_round_for_session") as generate,
        ):
            response = self.client.post(
                "/api/start-session",
                json={"mode": "classic", "category": "All Celebrities"},
            )
            session_id = response.json()["session_id"]
            guess_response = self.client.post(
                "/api/guess",
                json={"session_id": session_id, "guess": "alive"},
            )
            next_response = self.client.get(
                "/api/next-round",
                params={"session_id": session_id},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(guess_response.status_code, 200)
        self.assertEqual(next_response.status_code, 200)
        generate.assert_not_called()
        session = main.sessions[session_id]
        self.assertEqual(session["inventory_pack_id"], expected_pack_id)
        self.assertEqual(len(session["inventory_rounds"]), main.ROUNDS_PER_GAME - 2)
        self.assertEqual(session["active_round"]["person_name"], "All Celebrities Person 2")

    def test_enabled_empty_inventory_does_not_fall_back_to_live_generation(self) -> None:
        with (
            patch.object(main, "ROUND_INVENTORY_ENABLED", True),
            patch.object(main, "generate_round_for_session") as generate,
        ):
            response = self.client.post(
                "/api/start-session",
                json={"mode": "classic", "category": "All Celebrities"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("No prepared Classic game", response.json()["detail"])
        self.assertEqual(response.headers["retry-after"], "60")
        generate.assert_not_called()
        self.assertFalse(main.sessions)

    def test_finishing_reserved_classic_pack_marks_inventory_consumed(self) -> None:
        pack_id = self.stock_pack()
        session_id = str(uuid.uuid4())
        reservation = inventory.reserve_classic_pack(
            self.db_path,
            "All Celebrities",
            session_id,
        )
        session = app_session(round_number=main.ROUNDS_PER_GAME)
        session["inventory_pack_id"] = pack_id
        main.sessions[session_id] = session

        response = self.client.get("/api/next-round", params={"session_id": session_id})
        health = inventory.inventory_health(self.db_path)

        self.assertEqual(reservation["pack_id"], pack_id)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "finale")
        self.assertEqual(health["packs"]["consumed"], 1)
        self.assertEqual(health["rounds"]["consumed"], main.ROUNDS_PER_GAME)

    def test_status_reports_inventory_health_and_latency_percentiles(self) -> None:
        main.timing_telemetry.record("request.start_session", 10)
        main.timing_telemetry.record("request.start_session", 20)
        main.timing_telemetry.record("request.start_session", 90)

        with patch.object(main, "ROUND_INVENTORY_ENABLED", False):
            response = self.client.get("/api/status")
        payload = response.json()
        timing = payload["performance"]["timings"]["request.start_session"]

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["round_inventory_enabled"])
        self.assertEqual(
            payload["inventory_max_round_age_seconds"],
            main.INVENTORY_MAX_ROUND_AGE_SECONDS,
        )
        self.assertEqual(payload["inventory"]["schema_version"], inventory.SCHEMA_VERSION)
        self.assertEqual(payload["inventory"]["jobs"]["queued"], 0)
        self.assertEqual(payload["inventory"]["backlog_packs"], 0)
        self.assertEqual(timing["count"], 3)
        self.assertEqual(timing["p50_ms"], 20.0)
        self.assertEqual(timing["p95_ms"], 90.0)


if __name__ == "__main__":
    unittest.main()
