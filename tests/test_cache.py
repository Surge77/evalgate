from pathlib import Path

from evalgate.cache import ResponseCache, cache_key


def test_put_get_round_trip(tmp_path: Path) -> None:
    with ResponseCache(tmp_path / "cache.db") as cache:
        cache.put("k1", "value")
        assert cache.get("k1") == "value"


def test_miss_returns_none(tmp_path: Path) -> None:
    with ResponseCache(tmp_path / "cache.db") as cache:
        assert cache.get("absent") is None


def test_put_overwrites(tmp_path: Path) -> None:
    with ResponseCache(tmp_path / "cache.db") as cache:
        cache.put("k", "old")
        cache.put("k", "new")
        assert cache.get("k") == "new"


def test_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "cache.db"
    with ResponseCache(db) as cache:
        cache.put("k", "v")
    with ResponseCache(db) as reopened:
        assert reopened.get("k") == "v"


def test_creates_parent_directories(tmp_path: Path) -> None:
    with ResponseCache(tmp_path / "nested" / "dir" / "cache.db") as cache:
        cache.put("k", "v")
        assert cache.get("k") == "v"


def test_cache_key_stable_and_content_sensitive() -> None:
    assert cache_key("target", "a", "b") == cache_key("target", "a", "b")
    assert cache_key("target", "a", "b") != cache_key("target", "a", "c")
    assert cache_key("target", "ab") != cache_key("target", "a", "b")
    assert cache_key("judge", "a") != cache_key("target", "a")
