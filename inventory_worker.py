"""Restart-safe background replenishment worker for prepared game inventory."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import socket
import time

import inventory
from refill_inventory import (
    RoundGenerator,
    make_main_round_generator,
    make_main_round_revalidator,
)


logger = logging.getLogger(__name__)


class RefillLeaseLost(RuntimeError):
    pass


def process_one_job(
    db_path: Path,
    worker_id: str,
    generate_round: RoundGenerator,
    *,
    lease_ttl_seconds: int = 300,
    max_generated_rounds_per_pack: int = 30,
    retry_delay_seconds: int = 30,
    max_round_age_seconds: int = 86400,
) -> dict | None:
    """Claim and process one job, persisting enough progress for safe restart."""
    job = inventory.claim_refill_job(
        db_path,
        worker_id,
        lease_ttl_seconds=lease_ttl_seconds,
    )
    if job is None:
        return None
    job_id = int(job["id"])
    generated_rounds = 0
    created_pack_ids: list[int] = []
    forbidden = inventory.category_person_names(db_path, str(job["category"]))
    generation_limit = int(job["requested_packs"]) * max_generated_rounds_per_pack

    try:
        while inventory.count_job_created_packs(db_path, job_id) < int(
            job["requested_packs"]
        ):
            if not inventory.renew_refill_job_lease(
                db_path,
                job_id,
                worker_id,
                lease_ttl_seconds=lease_ttl_seconds,
            ):
                raise RefillLeaseLost(f"Lost lease for refill job {job_id}")
            pack_id = inventory.create_classic_pack(
                db_path,
                str(job["category"]),
                round_count=int(job["rounds_per_pack"]),
                alive_per_pack=int(job["alive_per_pack"]),
                source_job_id=job_id,
                max_round_age_seconds=max_round_age_seconds,
            )
            if pack_id is not None:
                created_pack_ids.append(pack_id)
                continue
            if generated_rounds >= generation_limit:
                raise RuntimeError(
                    f"Unable to assemble balanced packs after generating {generated_rounds} rounds"
                )
            model_name, payload = generate_round(forbidden, str(job["category"]))
            if not inventory.renew_refill_job_lease(
                db_path,
                job_id,
                worker_id,
                lease_ttl_seconds=lease_ttl_seconds,
            ):
                raise RefillLeaseLost(f"Lost lease for refill job {job_id} during generation")
            inventory.add_round(
                db_path,
                str(job["category"]),
                payload,
                source_model=model_name,
            )
            forbidden.append(str(payload["person_name"]))
            generated_rounds += 1

        if not inventory.complete_refill_job(db_path, job_id, worker_id):
            raise RefillLeaseLost(f"Lost lease before completing refill job {job_id}")
        return {
            "job_id": job_id,
            "state": "completed",
            "generated_rounds": generated_rounds,
            "created_pack_ids": created_pack_ids,
            "total_job_packs": inventory.count_job_created_packs(db_path, job_id),
        }
    except Exception as exc:
        next_state = inventory.fail_refill_job(
            db_path,
            job_id,
            worker_id,
            str(exc),
            retry_delay_seconds=retry_delay_seconds,
        )
        logger.error("Refill job %s failed: %s", job_id, exc)
        return {
            "job_id": job_id,
            "state": next_state or "lease_lost",
            "generated_rounds": generated_rounds,
            "created_pack_ids": created_pack_ids,
            "error": str(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maintain prepared Classic game inventory.")
    parser.add_argument("--db", type=Path)
    parser.add_argument("--category", default="All Celebrities")
    parser.add_argument("--minimum-packs", type=int, default=2)
    parser.add_argument("--rounds-per-pack", type=int, default=10)
    parser.add_argument("--alive-per-pack", type=int, default=5)
    parser.add_argument("--lease-seconds", type=int, default=300)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-generated-rounds-per-pack", type=int, default=30)
    parser.add_argument("--max-round-age-seconds", type=int, default=86400)
    parser.add_argument("--revalidation-limit", type=int, default=100)
    parser.add_argument("--failed-job-cooldown-seconds", type=int, default=3600)
    parser.add_argument("--worker-id")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.db is not None:
        resolved_db = args.db.expanduser().resolve()
        os.environ["LEADERBOARD_DB_FILE"] = str(resolved_db)

    import main as app_main

    db_path = args.db.expanduser().resolve() if args.db is not None else app_main.DB_PATH
    app_main.DB_PATH = db_path
    app_main.init_db()
    category = app_main.StartSessionRequest(
        mode="classic",
        category=args.category,
    ).category
    worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"
    generator = make_main_round_generator(app_main)
    revalidator = make_main_round_revalidator(app_main)

    while True:
        retired_packs = inventory.retire_stale_available_packs(
            db_path,
            max_round_age_seconds=args.max_round_age_seconds,
        )
        revalidation = inventory.revalidate_stale_available_rounds(
            db_path,
            revalidator,
            max_round_age_seconds=args.max_round_age_seconds,
            limit=args.revalidation_limit,
        )
        plan = inventory.ensure_minimum_pack_stock(
            db_path,
            category,
            minimum_ready_packs=args.minimum_packs,
            rounds_per_pack=args.rounds_per_pack,
            alive_per_pack=args.alive_per_pack,
            failed_job_cooldown_seconds=args.failed_job_cooldown_seconds,
        )
        result = process_one_job(
            db_path,
            worker_id,
            generator,
            lease_ttl_seconds=args.lease_seconds,
            max_generated_rounds_per_pack=args.max_generated_rounds_per_pack,
            max_round_age_seconds=args.max_round_age_seconds,
        )
        if args.once:
            print(
                json.dumps(
                    {
                        "maintenance": {
                            "retired_packs": retired_packs,
                            "revalidation": revalidation,
                        },
                        "plan": plan,
                        "result": result,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if result is None or result["state"] == "completed" else 2
        if result is None:
            time.sleep(max(0.25, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
