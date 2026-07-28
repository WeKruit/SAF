from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pandas as pd
import pytest

from prediction_market.immutable_object_store import (
    ArtifactRefV1,
    content_addressed_key,
)
from prediction_market.sports.nfl_factor_lab_bundles import (
    build_cross_game_bundle,
    build_single_game_bundle,
)
from prediction_market.sports.nfl_factor_lab_run_index import (
    build_factor_lab_run_index,
    factor_lab_materializer_code_lock_paths,
    factor_lab_materializer_code_lock_sha256,
)
from prediction_market.sports.nfl_factor_lab_storage import (
    FactorLabStorageError,
    FactorLabStorageRefsV2,
    build_factor_lab_storage_refs,
    configured_factor_lab_object_store,
    factor_lab_storage_status,
    hydrate_factor_lab_ref,
    mirror_factor_lab_storage,
    verify_factor_lab_remote_inventory,
)
from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


class MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.put_count = 0

    def head_object(self, key: str) -> dict[str, object] | None:
        payload = self.objects.get(key)
        if payload is None:
            return None
        return {
            "byte_size": len(payload),
            "metadata": dict(self.metadata[key]),
        }

    def put_file(
        self,
        key: str,
        path: Path,
        *,
        metadata: dict[str, str],
    ) -> None:
        self.put_count += 1
        self.objects[key] = path.read_bytes()
        self.metadata[key] = dict(metadata)

    def read_object_chunks(
        self,
        key: str,
        *,
        chunk_size: int = 1024 * 1024,
    ):
        payload = self.objects[key]
        for offset in range(0, len(payload), chunk_size):
            yield payload[offset : offset + chunk_size]


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _single_frames(game_id: str) -> dict[str, pd.DataFrame]:
    grains: dict[str, dict[str, object]] = {
        "sports_row_disposition": {
            "raw_play_id": "1",
            "canonical_play_id": "1",
            "episode_id": "e1",
        },
        "canonical_events": {"play_id": "1"},
        "finalized_episodes": {"episode_id": "e1"},
        "stat_ledger": {"through_play_id": "1"},
        "pit_features": {"play_id": "1"},
        "contract_inventory": {
            "logical_market_id": "winner",
            "outcome": "HOME",
        },
        "market_observations": {
            "observation_id": "o1",
            "logical_market_id": "winner",
            "outcome": "HOME",
        },
        "market_cleaning_audit": {
            "audit_row_id": "m1",
            "observation_id": "o1",
            "included": True,
        },
        "game_time_intervals": {
            "interval_id": "g1",
            "episode_id": "e1",
            "delay_seconds": 0,
        },
        "market_time_intervals": {
            "interval_id": "m1",
            "observation_id": "o1",
        },
        "source_timeline": {
            "timeline_id": "t1",
            "interval_id": "g1",
            "layer": "G",
            "delay_seconds": 0,
        },
        "timeline_audit": {
            "timeline_audit_id": "ta1",
            "timeline_id": "t1",
        },
        "association_attrition": {
            "association_id": "a1",
            "episode_id": "e1",
            "logical_market_id": "winner",
            "outcome": "HOME",
        },
        "reaction_paths": {
            "episode_id": "e1",
            "logical_market_id": "winner",
            "outcome": "HOME",
            "venue": "polymarket",
            "market_family": "moneyline",
            "horizon_seconds": 30,
        },
        "factor_observations": {
            "factor_id": "NFL.EVENT.PASS",
            "episode_id": "e1",
            "logical_market_id": "winner",
            "outcome": "HOME",
            "venue": "polymarket",
            "market_family": "moneyline",
            "horizon_seconds": 30,
        },
        "factor_exclusion_audit": {"audit_id": "f1"},
        "single_game_factor_results": {
            "factor_id": "NFL.EVENT.PASS",
            "market_family": "moneyline",
            "horizon_seconds": 30,
        },
    }
    return {
        name: pd.DataFrame([{"game_id": game_id, **row}])
        for name, row in grains.items()
    }


def _cross_frames() -> dict[str, pd.DataFrame]:
    factor_key = {
        "factor_id": "NFL.EVENT.PASS",
        "market_family": "moneyline",
        "horizon_seconds": 30,
    }
    per_game = [
        {**factor_key, "game_id": game_id} for game_id in X13_GAME_IDS
    ]
    return {
        "cross_game_factor_results": pd.DataFrame([factor_key]),
        "game_contribution": pd.DataFrame(per_game),
        "leave_one_game_out": pd.DataFrame(per_game),
        "venue_consistency": pd.DataFrame(
            [{**factor_key, "venue": "polymarket"}]
        ),
        "expert_review_queue": pd.DataFrame(
            [{**factor_key, "decision": "PENDING"}]
        ),
    }


def _verified_run_index(tmp_path: Path):
    for relative in factor_lab_materializer_code_lock_paths():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    authoritative_code_lock = factor_lab_materializer_code_lock_sha256(
        tmp_path
    )
    singles = tuple(
        build_single_game_bundle(
            game_id=game_id,
            stage_frames=_single_frames(game_id),
            source_hashes=(HASH_A,),
            factor_registry_sha256=HASH_B,
            code_lock_sha256=authoritative_code_lock,
            output_root=tmp_path / "single",
        )
        for game_id in X13_GAME_IDS
    )
    cross = build_cross_game_bundle(
        single_game_manifests=tuple(
            bundle.manifest_path for bundle in singles
        ),
        cross_game_frames=_cross_frames(),
        batch_spec_sha256=HASH_A,
        factor_registry_sha256=HASH_B,
        code_lock_sha256=authoritative_code_lock,
        output_root=tmp_path / "cross",
    )
    source = tmp_path / "frozen-source.json"
    source.write_bytes(b'{"frozen":true}\n')
    return build_factor_lab_run_index(
        single_game_manifests=tuple(
            bundle.manifest_path for bundle in singles
        ),
        cross_game_manifest=cross.manifest_path,
        source_artifacts={"raw_sports": source},
        project_root=tmp_path,
        s3_bucket="prediction-market-data",
        s3_prefix="prediction-market",
        output_path=tmp_path / "run-index.json",
    )


def _ref(
    root: Path,
    *,
    relative: str,
    namespace: str,
    payload: bytes,
) -> ArtifactRefV1:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = _digest(payload)
    return ArtifactRefV1(
        sha256=digest,
        byte_length=len(payload),
        media_type="application/octet-stream",
        local_relative_path=relative,
        s3_bucket="prediction-market-data",
        s3_key=content_addressed_key(
            prefix="prediction-market",
            namespace=namespace,
            sha256=digest,
            suffix=".bin",
        ),
    )


def _three_scope_refs(root: Path) -> FactorLabStorageRefsV2:
    return FactorLabStorageRefsV2(
        run_index_path=root / "run-index.json",
        run_index_sha256=_digest(b"run-index"),
        s3_bucket="prediction-market-data",
        s3_prefix="prediction-market",
        single_game_refs=(
            _ref(
                root,
                relative="local/single.bin",
                namespace="research/nfl/x13/single-game/game",
                payload=b"single",
            ),
        ),
        cross_game_refs=(
            _ref(
                root,
                relative="local/cross.bin",
                namespace="research/nfl/x13/cross-game",
                payload=b"cross",
            ),
        ),
        source_refs=(
            _ref(
                root,
                relative="local/source.bin",
                namespace="raw/raw_sports",
                payload=b"source",
            ),
        ),
    )


def _parallel_refs(
    root: Path,
    *,
    reverse: bool = False,
) -> FactorLabStorageRefsV2:
    single_refs = tuple(
        _ref(
            root,
            relative=f"local/single-{index}.bin",
            namespace=f"research/nfl/x13/single-game/game-{index}",
            payload=f"single-{index}".encode(),
        )
        for index in range(4)
    )
    if reverse:
        single_refs = tuple(reversed(single_refs))
    return FactorLabStorageRefsV2(
        run_index_path=root / "run-index.json",
        run_index_sha256=_digest(b"run-index"),
        s3_bucket="prediction-market-data",
        s3_prefix="prediction-market",
        single_game_refs=single_refs,
        cross_game_refs=(
            _ref(
                root,
                relative="local/cross-parallel.bin",
                namespace="research/nfl/x13/cross-game",
                payload=b"cross-parallel",
            ),
        ),
        source_refs=(
            _ref(
                root,
                relative="local/source-parallel.bin",
                namespace="raw/raw_sports",
                payload=b"source-parallel",
            ),
        ),
    )


class ConcurrentReadbackStore(MemoryStore):
    def __init__(self, tracked_keys: set[str]) -> None:
        super().__init__()
        self._tracked_keys = tracked_keys
        self._barrier = threading.Barrier(4, timeout=3)
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def read_object_chunks(
        self,
        key: str,
        *,
        chunk_size: int = 1024 * 1024,
    ):
        tracked = key in self._tracked_keys
        if tracked:
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                self._barrier.wait()
                yield from super().read_object_chunks(
                    key, chunk_size=chunk_size
                )
            finally:
                with self._lock:
                    self.active -= 1
            return
        yield from super().read_object_chunks(key, chunk_size=chunk_size)


class FailingReadbackStore(MemoryStore):
    def __init__(self, failing_key: str) -> None:
        super().__init__()
        self._failing_key = failing_key

    def read_object_chunks(
        self,
        key: str,
        *,
        chunk_size: int = 1024 * 1024,
    ):
        if key == self._failing_key:
            raise OSError("deliberate readback failure")
        yield from super().read_object_chunks(key, chunk_size=chunk_size)


def test_build_refs_covers_verified_single_cross_and_source_objects(
    tmp_path: Path,
) -> None:
    run_index = _verified_run_index(tmp_path)

    refs = build_factor_lab_storage_refs(
        run_index,
        project_root=tmp_path,
    )

    assert len(refs.single_game_refs) == len(X13_GAME_IDS) * 18
    assert len(refs.cross_game_refs) == 6
    assert len(refs.source_refs) == 1
    assert refs.s3_prefix == "prediction-market"
    assert all(
        ref.s3_bucket == "prediction-market-data" for ref in refs.all_refs
    )
    assert refs.source_refs[0].s3_key.startswith(
        "prediction-market/raw/raw_sports/objects/sha256/"
    )
    assert all(
        "/research/nfl/x13/research/nfl/x13/" not in ref.s3_key
        for ref in refs.all_refs
    )


def test_mirror_writes_deterministic_readback_verified_inventory(
    tmp_path: Path,
) -> None:
    refs = _three_scope_refs(tmp_path)
    refs.run_index_path.write_bytes(b"run-index")
    store = MemoryStore()

    first = mirror_factor_lab_storage(
        refs,
        project_root=tmp_path,
        inventory_root=tmp_path / "inventory",
        object_store=store,
    )
    first_put_count = store.put_count
    second = mirror_factor_lab_storage(
        refs,
        project_root=tmp_path,
        inventory_root=tmp_path / "inventory",
        object_store=store,
    )
    verified = verify_factor_lab_remote_inventory(
        first.path,
        project_root=tmp_path,
        expected_refs=refs,
    )

    assert first == second == verified
    assert len(verified.object_refs) == 3
    assert verified.inventory_ref.s3_key in store.objects
    assert store.put_count == first_put_count
    assert verified.total_byte_length == sum(
        ref.byte_length for ref in refs.all_refs
    )
    assert all(
        store.metadata[ref.s3_key]["sha256"] == ref.sha256
        for ref in (*refs.all_refs, verified.inventory_ref)
    )


def test_mirror_runs_four_readbacks_concurrently_and_keeps_fixed_order(
    tmp_path: Path,
) -> None:
    refs = _parallel_refs(tmp_path)
    refs.run_index_path.write_bytes(b"run-index")
    tracked = {ref.s3_key for ref in refs.single_game_refs}
    store = ConcurrentReadbackStore(tracked)

    first = mirror_factor_lab_storage(
        refs,
        project_root=tmp_path,
        inventory_root=tmp_path / "inventory",
        object_store=store,
        max_workers=4,
    )
    reversed_refs = _parallel_refs(tmp_path, reverse=True)
    second = mirror_factor_lab_storage(
        reversed_refs,
        project_root=tmp_path,
        inventory_root=tmp_path / "inventory",
        object_store=store,
        max_workers=4,
    )

    assert store.max_active >= 4
    assert first == second
    assert first.path.read_bytes() == second.path.read_bytes()
    assert tuple(ref.s3_key for ref in first.object_refs) == tuple(
        ref.s3_key for ref in second.object_refs
    )


def test_parallel_failure_never_publishes_remote_inventory(
    tmp_path: Path,
) -> None:
    refs = _parallel_refs(tmp_path)
    refs.run_index_path.write_bytes(b"run-index")
    failing_key = refs.single_game_refs[1].s3_key
    store = FailingReadbackStore(failing_key)

    with pytest.raises(FactorLabStorageError, match="readback"):
        mirror_factor_lab_storage(
            refs,
            project_root=tmp_path,
            inventory_root=tmp_path / "inventory",
            object_store=store,
            max_workers=4,
        )

    assert not (tmp_path / "inventory").exists()
    assert all(
        "/remote-inventory/" not in key for key in store.objects
    )


def test_hydrate_uses_only_ref_and_fails_closed_for_missing_or_bad_remote(
    tmp_path: Path,
) -> None:
    refs = _three_scope_refs(tmp_path)
    ref = refs.source_refs[0]
    payload = (tmp_path / ref.local_relative_path).read_bytes()
    store = MemoryStore()
    store.objects[ref.s3_key] = payload
    store.metadata[ref.s3_key] = {
        "sha256": ref.sha256,
        "byte_length": str(ref.byte_length),
        "media_type": ref.media_type,
    }

    hydrated = hydrate_factor_lab_ref(
        ref,
        object_store=store,
        cache_root=tmp_path / "cache",
    )
    assert hydrated.read_bytes() == payload

    hydrated.unlink()
    store.objects.pop(ref.s3_key)
    store.metadata.pop(ref.s3_key)
    with pytest.raises(FactorLabStorageError, match="remote object is missing"):
        hydrate_factor_lab_ref(
            ref,
            object_store=store,
            cache_root=tmp_path / "cache",
        )

    store.objects[ref.s3_key] = b"tampered"
    store.metadata[ref.s3_key] = {"sha256": ref.sha256}
    with pytest.raises(FactorLabStorageError, match="remote"):
        hydrate_factor_lab_ref(
            ref,
            object_store=store,
            cache_root=tmp_path / "cache",
        )


def test_status_reports_verified_missing_and_conflicting_objects(
    tmp_path: Path,
) -> None:
    refs = _three_scope_refs(tmp_path)
    store = MemoryStore()
    single, cross, source = (
        refs.single_game_refs[0],
        refs.cross_game_refs[0],
        refs.source_refs[0],
    )
    for ref in (single, cross):
        payload = (tmp_path / ref.local_relative_path).read_bytes()
        store.objects[ref.s3_key] = payload
        store.metadata[ref.s3_key] = {
            "sha256": ref.sha256,
            "byte_length": str(ref.byte_length),
            "media_type": ref.media_type,
        }
    (tmp_path / cross.local_relative_path).write_bytes(b"tampered-local")
    (tmp_path / source.local_relative_path).unlink()

    status = factor_lab_storage_status(
        refs,
        project_root=tmp_path,
        object_store=store,
    )
    by_key = {row.ref.s3_key: row for row in status.rows}

    assert by_key[single.s3_key].state == "LOCAL_REMOTE_VERIFIED"
    assert by_key[cross.s3_key].state == "CONFLICT"
    assert by_key[source.s3_key].state == "MISSING"
    assert status.local_verified_count == 1
    assert status.remote_verified_count == 2
    assert status.conflict_count == 1


def test_inventory_tampering_is_rejected(tmp_path: Path) -> None:
    refs = _three_scope_refs(tmp_path)
    refs.run_index_path.write_bytes(b"run-index")
    verified = mirror_factor_lab_storage(
        refs,
        project_root=tmp_path,
        inventory_root=tmp_path / "inventory",
        object_store=MemoryStore(),
    )
    verified.path.write_bytes(verified.path.read_bytes() + b" ")

    with pytest.raises(FactorLabStorageError, match="canonical"):
        verify_factor_lab_remote_inventory(
            verified.path,
            project_root=tmp_path,
            expected_refs=refs,
        )


def test_configured_store_rejects_noncanonical_base_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore()

    monkeypatch.setattr(
        "prediction_market.sports.nfl_factor_lab_storage.configured_object_store",
        lambda **_kwargs: (store, "other-prefix"),
    )

    with pytest.raises(FactorLabStorageError, match="prediction-market"):
        configured_factor_lab_object_store()
