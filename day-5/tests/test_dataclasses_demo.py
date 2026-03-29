"""Tests for dataclass functions."""

import pytest

from src.dataclasses_demo import Config, FileStats, Point, Rectangle


class TestPointDataclass:
    """Test Point dataclass."""

    def test_point_creation_happy_path(self) -> None:
        """Test creating a point."""
        p = Point(3.0, 4.0)
        assert p.x == pytest.approx(3.0)
        assert p.y == pytest.approx(4.0)

    def test_point_distance_from_origin(self) -> None:
        """Test distance calculation."""
        p = Point(3.0, 4.0)
        dist = p.distance_from_origin()
        assert dist == pytest.approx(5.0)

    def test_point_invalid_type(self) -> None:
        """Test error on invalid type."""
        with pytest.raises(TypeError):
            Point("3", 4)  # type: ignore

    def test_point_equality(self) -> None:
        """Test dataclass equality."""
        p1 = Point(1.0, 2.0)
        p2 = Point(1.0, 2.0)
        assert p1 == p2

    def test_point_repr(self) -> None:
        """Test dataclass repr."""
        p = Point(1.0, 2.0)
        repr_str = repr(p)
        assert "Point" in repr_str
        assert "1.0" in repr_str


class TestRectangleDataclass:
    """Test Rectangle dataclass."""

    def test_rectangle_creation_happy_path(self) -> None:
        """Test creating a rectangle."""
        p = Point(0.0, 0.0)
        r = Rectangle(p, 5.0, 10.0)
        assert r.width == pytest.approx(5.0)
        assert r.height == pytest.approx(10.0)

    def test_rectangle_area(self) -> None:
        """Test area calculation."""
        p = Point(0.0, 0.0)
        r = Rectangle(p, 5.0, 10.0)
        assert r.area() == pytest.approx(50.0)

    def test_rectangle_invalid_dimensions(self) -> None:
        """Test error on invalid dimensions."""
        p = Point(0.0, 0.0)
        with pytest.raises(ValueError):
            Rectangle(p, -5.0, 10.0)


class TestImmutableConfig:
    """Test frozen dataclass."""

    def test_config_creation_happy_path(self) -> None:
        """Test creating config."""
        c = Config("prod", 30)
        assert c.name == "prod"
        assert c.timeout == 30

    def test_config_immutable(self) -> None:
        """Test that config is immutable."""
        c = Config("prod", 30)
        with pytest.raises(Exception):  # FrozenInstanceError
            c.name = "dev"  # type: ignore

    def test_config_invalid_timeout(self) -> None:
        """Test error on invalid timeout."""
        with pytest.raises(ValueError):
            Config("prod", -1)


class TestFileStats:
    """Test FileStats dataclass."""

    def test_file_stats_creation_happy_path(self) -> None:
        """Test creating file stats."""
        fs = FileStats("file.py", 100, 5000, 800)
        assert fs.line_count == 100
        assert fs.char_count == 5000

    def test_file_stats_with_imports(self) -> None:
        """Test file stats with imports."""
        fs = FileStats(
            "file.py", 50, 2000, 400, ["os", "sys"]
        )
        assert len(fs.imports) == 2

    def test_file_stats_summary(self) -> None:
        """Test summary generation."""
        fs = FileStats("file.py", 10, 500, 100, ["os"])
        summary = fs.summary()
        assert "file.py" in summary
        assert "10 lines" in summary

    def test_file_stats_negative_counts(self) -> None:
        """Test error on negative counts."""
        with pytest.raises(ValueError):
            FileStats("file.py", -1, 500, 100)