"""Offline producer for validated Alive or Dead Classic game packs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Callable

import inventory


RoundGenerator = Callable[[list[str], str], tuple[str, dict]]


def make_main_round_generator(app_main) -> RoundGenerator:
    """Build the production generator while keeping worker tests dependency-free."""
    def generate_round(forbidden: list[str], category: str) -> tuple[str, dict]:
        started_at = app_main.perf_now()
        try:
            model_name, payload = app_main.generate_single_round_sync(
                forbidden,
                app_main.BACKGROUND_MODEL_CANDIDATES,
                False,
                category=category,
            )
        except Exception:
            app_main.log_perf("generation.external", started_at, outcome="error", source="refill")
            raise
        app_main.log_perf(
            "generation.external",
            started_at,
            outcome="success",
            source="refill",
            model=model_name,
        )
        facts = app_main.verify_person_facts(payload["person_name"])
        enriched = dict(payload)
        enriched.update(
            {
                "wikidata_entity_id": facts.wikidata_entity_id,
                "wikipedia_title": facts.wikipedia_title,
                "birth_date": facts.birth_date,
                "portrait_url": app_main.get_round_portrait_url(payload),
            }
        )
        return model_name, enriched

    return generate_round


def make_main_round_revalidator(app_main):
    """Recheck identity, vital status, birth date, and the embedded portrait."""
    def revalidate(payload: dict) -> dict:
        facts = app_main.verify_person_facts(payload["person_name"], use_cache=False)
        portrait_url = app_main.get_round_portrait_url(payload)
        if not portrait_url:
            raise ValueError("Stored round has no portrait URL")
        app_main.validate_image_url_reachable(
            portrait_url,
            payload["person_name"],
            "Inventory revalidation",
            use_cache=False,
        )
        return {
            "actual_status": facts.actual_status,
            "wikidata_entity_id": facts.wikidata_entity_id,
            "wikipedia_title": facts.wikipedia_title,
            "birth_date": facts.birth_date,
            "portrait_url": portrait_url,
        }

    return revalidate


def refill_inventory(
    db_path: Path,
    category: str,
    *,
    rounds_to_generate: int,
    packs_to_create: int,
    rounds_per_pack: int,
    generate_round: RoundGenerator,
) -> dict:
    if rounds_to_generate < 0 or packs_to_create < 0 or rounds_per_pack < 1:
        raise ValueError("Round and pack counts must be non-negative; pack size must be positive")
    forbidden = inventory.category_person_names(db_path, category)
    inserted_round_ids: list[int] = []
    generated_models: list[str] = []
    for _ in range(rounds_to_generate):
        model_name, payload = generate_round(forbidden, category)
        round_id = inventory.add_round(
            db_path,
            category,
            payload,
            source_model=model_name,
        )
        inserted_round_ids.append(round_id)
        generated_models.append(model_name)
        forbidden.append(str(payload["person_name"]))

    pack_ids: list[int] = []
    for _ in range(packs_to_create):
        pack_id = inventory.create_classic_pack(
            db_path,
            category,
            round_count=rounds_per_pack,
        )
        if pack_id is None:
            break
        pack_ids.append(pack_id)

    return {
        "category": " ".join(category.split()),
        "generated_rounds": len(inserted_round_ids),
        "round_ids": inserted_round_ids,
        "source_models": sorted(set(generated_models)),
        "created_packs": len(pack_ids),
        "pack_ids": pack_ids,
        "inventory": inventory.inventory_health(db_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate validated rounds offline and assemble ready Classic packs."
    )
    parser.add_argument("--category", default="All Celebrities")
    parser.add_argument("--packs", type=int, default=1)
    parser.add_argument("--rounds-per-pack", type=int, default=10)
    parser.add_argument(
        "--rounds",
        type=int,
        help="Rounds to generate before packing (defaults to packs × rounds-per-pack).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Inventory database path (defaults to LEADERBOARD_DB_FILE/leaderboard.db).",
    )
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
    rounds_to_generate = (
        args.rounds
        if args.rounds is not None
        else args.packs * args.rounds_per_pack
    )

    generate_round = make_main_round_generator(app_main)

    result = refill_inventory(
        db_path,
        args.category,
        rounds_to_generate=rounds_to_generate,
        packs_to_create=args.packs,
        rounds_per_pack=args.rounds_per_pack,
        generate_round=generate_round,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["created_packs"] != args.packs:
        print(
            "Not enough unpacked validated rounds were available to create every requested pack.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
