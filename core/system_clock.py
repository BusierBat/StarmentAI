"""
StarmentAI - System Clock
Provides real-time date, time, and timezone information from the operating system.
"""

import datetime
import time
from dataclasses import dataclass
from typing import Optional

try:
    import tzlocal
    HAS_TZLOCAL = True
except ImportError:
    HAS_TZLOCAL = False

try:
    from zoneinfo import ZoneInfo
    HAS_ZONEINFO = True
except ImportError:  # pragma: no cover - zoneinfo is stdlib on 3.9+
    HAS_ZONEINFO = False


class SystemClockError(Exception):
    """Base exception for system clock errors."""


@dataclass(frozen=True, slots=True)
class DateTimeInfo:
    """Structured datetime information from the system."""
    iso_datetime: str
    date: str
    time: str
    timezone: str
    utc_offset: str
    weekday: str
    timestamp: float

    def format_for_prompt(self) -> str:
        """Format as a block for system prompt injection."""
        return (
            f"CURRENT_DATETIME: {self.iso_datetime}\n"
            f"TIMEZONE: {self.timezone}\n"
            f"UTC_OFFSET: {self.utc_offset}"
        )

    def format_short(self) -> str:
        """Short human-readable format."""
        return f"{self.date} {self.time} ({self.timezone}, UTC{self.utc_offset})"


class SystemClock:
    """Retrieves current date, time, and timezone from the operating system."""

    def __init__(self, default_timezone: str = "UTC") -> None:
        self.default_timezone = default_timezone
        self._tz_cache: Optional[str] = None

    def _get_local_timezone(self) -> str:
        """Get the local timezone name, with caching."""
        if self._tz_cache is not None:
            return self._tz_cache

        if HAS_TZLOCAL:
            try:
                tz = tzlocal.get_localzone()
                self._tz_cache = str(tz)
                return self._tz_cache
            except Exception:
                pass

        # Fallback: try to detect from time.tzname
        try:
            if time.tzname[0] != time.tzname[1]:
                # Has DST
                is_dst = time.localtime().tm_isdst > 0
                self._tz_cache = time.tzname[1 if is_dst else 0]
            else:
                self._tz_cache = time.tzname[0]
            return self._tz_cache
        except Exception:
            self._tz_cache = self.default_timezone
            return self._tz_cache

    def _get_utc_offset(self, dt: datetime.datetime) -> str:
        """Get UTC offset in ±HH:MM format."""
        try:
            if dt.tzinfo is not None:
                offset = dt.tzinfo.utcoffset(dt)
                if offset is not None:
                    total_seconds = int(offset.total_seconds())
                    sign = "+" if total_seconds >= 0 else "-"
                    hours = abs(total_seconds) // 3600
                    minutes = (abs(total_seconds) % 3600) // 60
                    return f"{sign}{hours:02d}:{minutes:02d}"
        except Exception:
            pass
        return "+00:00"

    def _resolve_timezone_name(
        self,
        detected: str,
        aware_dt: datetime.datetime | None,
    ) -> str:
        """
        Prefer the configured timezone name when its offset matches local time.

        Some platforms (notably Windows via tzlocal) report a coarse zone
        such as ``Asia/Bangkok`` for UTC+7. When the configured zone
        (e.g. ``Asia/Ho_Chi_Minh``) has the same UTC offset at the current
        instant, we use the configured name so the reported timezone is the
        one the operator expects.
        """
        if (
            not HAS_ZONEINFO
            or not self.default_timezone
            or self.default_timezone == "UTC"
            or aware_dt is None
        ):
            return detected

        try:
            preferred = ZoneInfo(self.default_timezone)
            local_offset = aware_dt.utcoffset()
            preferred_offset = aware_dt.astimezone(preferred).utcoffset()
            if (
                local_offset is not None
                and preferred_offset is not None
                and local_offset == preferred_offset
            ):
                return self.default_timezone
        except Exception:
            pass

        return detected

    def now(self) -> DateTimeInfo:
        """
        Get current datetime information from the system.

        Returns:
            DateTimeInfo with current date, time, timezone, and UTC offset.

        Raises:
            SystemClockError: If unable to determine system time.
        """
        try:
            # Get current time with timezone awareness
            now = datetime.datetime.now()
            timestamp = now.timestamp()

            # Get timezone name
            timezone = self._get_local_timezone()

            # Make timezone-aware for offset calculation
            try:
                if HAS_TZLOCAL:
                    tz = tzlocal.get_localzone()
                    aware_dt = now.replace(tzinfo=tz)
                else:
                    aware_dt = now.astimezone()
            except Exception:
                aware_dt = now

            utc_offset = self._get_utc_offset(aware_dt)
            timezone = self._resolve_timezone_name(timezone, aware_dt)

            # Format components
            iso_datetime = aware_dt.isoformat()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            weekday = now.strftime("%A")

            return DateTimeInfo(
                iso_datetime=iso_datetime,
                date=date_str,
                time=time_str,
                timezone=timezone,
                utc_offset=utc_offset,
                weekday=weekday,
                timestamp=timestamp,
            )

        except Exception as exc:
            raise SystemClockError(f"Failed to get system time: {exc}") from exc

    def get_timezone_info(self) -> dict[str, str]:
        """Get timezone information without full datetime."""
        tz = self._get_local_timezone()
        now = datetime.datetime.now()
        try:
            if HAS_TZLOCAL:
                tz_obj = tzlocal.get_localzone()
                aware = now.replace(tzinfo=tz_obj)
            else:
                aware = now.astimezone()
            offset = self._get_utc_offset(aware)
            tz = self._resolve_timezone_name(tz, aware)
        except Exception:
            offset = "+00:00"
        return {"timezone": tz, "utc_offset": offset}

    def format_for_model(self) -> str:
        """Get formatted datetime block for model context."""
        try:
            return self.now().format_for_prompt()
        except SystemClockError:
            return (
                "CURRENT_DATETIME: UNAVAILABLE\n"
                "TIMEZONE: UNKNOWN\n"
                "UTC_OFFSET: UNKNOWN\n"
                "(Không thể xác định thời gian hệ thống lúc này; "
                "không tự đoán ngày/giờ.)"
            )


# Global instance for convenience
_default_clock: Optional[SystemClock] = None


def get_system_clock(default_timezone: str | None = None) -> SystemClock:
    """Get the default system clock instance."""
    global _default_clock
    if _default_clock is None:
        _default_clock = SystemClock(
            default_timezone=default_timezone or "UTC"
        )
    return _default_clock


def get_current_datetime() -> DateTimeInfo:
    """Convenience function to get current datetime info."""
    return get_system_clock().now()


def format_datetime_for_prompt() -> str:
    """Convenience function to get formatted datetime for prompt."""
    return get_system_clock().format_for_model()