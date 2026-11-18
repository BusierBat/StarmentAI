import os
import sys
import time
import threading
import re
import unicodedata
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


HELP_TEXT = (
    "Các lệnh hỗ trợ:\n"
    "  /help    Hiển thị hướng dẫn\n"
    "  /clear   Xóa lịch sử hội thoại\n"
    "  /exit    Thoát chương trình\n"
    "Mẹo: dùng phím mũi tên để cuộn tin nhắn cũ, "
    "PageUp/PageDown để cuộn nhanh."
)

SPECIAL_KEYS = {
    "G": "home",
    "H": "up",
    "I": "pageup",
    "K": "left",
    "M": "right",
    "O": "end",
    "P": "down",
    "Q": "pagedown",
    "R": "insert",
    "S": "delete",
}


def char_width(ch):
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def strip_ansi(value):
    return ANSI_RE.sub("", value)


def text_width(value):
    return sum(char_width(ch) for ch in strip_ansi(value))


def pad_right(value, width):
    return value + " " * max(0, width - text_width(value))


def pad_left(value, width):
    return " " * max(0, width - text_width(value)) + value


def pad_center(value, width):
    total = text_width(value)
    left = max(0, (width - total) // 2)
    right = max(0, width - total - left)
    return " " * left + value + " " * right


def truncate_text(value, width):
    if width <= 0:
        return ""
    out = ""
    used = 0
    for ch in value:
        cw = char_width(ch)
        if used + cw > width:
            break
        out += ch
        used += cw
    return out


def split_long_word(word, width):
    if width <= 0:
        return [word]
    parts = []
    current = ""
    current_w = 0
    for ch in word:
        cw = char_width(ch)
        if current and current_w + cw > width:
            parts.append(current)
            current = ch
            current_w = cw
        else:
            current += ch
            current_w += cw
    parts.append(current)
    return parts


def wrap_text(text, width):
    if width <= 0:
        return [""]
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
            continue
        words = paragraph.split(" ")
        line = ""
        line_w = 0
        for word in words:
            if word == "":
                continue
            chunks = (
                split_long_word(word, width)
                if text_width(word) > width
                else [word]
            )
            for index, chunk in enumerate(chunks):
                chunk_w = text_width(chunk)
                space = 1 if (line and index == 0) else 0
                if line and line_w + space + chunk_w > width:
                    lines.append(line)
                    line = chunk
                    line_w = chunk_w
                else:
                    line = line + (" " if (line and index == 0) else "") + chunk
                    line_w += space + chunk_w
        lines.append(line)
    return lines


def gradient_text(text: str, start_rgb: Tuple[int, int, int], end_rgb: Tuple[int, int, int]) -> str:
    """Apply a horizontal gradient to text using true color ANSI."""
    stripped = strip_ansi(text)
    if not stripped:
        return text
    chars = list(stripped)
    n = len(chars)
    if n <= 1:
        r, g, b = start_rgb
        return f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m"
    result = []
    for i, ch in enumerate(chars):
        ratio = i / (n - 1)
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
        result.append(f"\x1b[38;2;{r};{g};{b}m{ch}")
    result.append("\x1b[0m")
    return "".join(result)


def gradient_text_vertical(lines: List[str], start_rgb: Tuple[int, int, int], end_rgb: Tuple[int, int, int]) -> List[str]:
    """Apply a vertical gradient across multiple lines."""
    n = len(lines)
    if n <= 1:
        return [gradient_text(l, start_rgb, end_rgb) for l in lines]
    result = []
    for i, line in enumerate(lines):
        ratio = i / (n - 1)
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
        result.append(gradient_text(line, (r, g, b), end_rgb))
    return result


class Palette:
    def __init__(self, mode="256"):
        self.mode = mode
        self.reset = "\x1b[0m" if mode != "none" else ""
        self.bold = "\x1b[1m" if mode != "none" else ""
        self.bold_off = "\x1b[22m" if mode != "none" else ""
        self.dim = "\x1b[2m" if mode != "none" else ""
        self.dim_off = "\x1b[22m" if mode != "none" else ""
        self.italic = "\x1b[3m" if mode != "none" else ""
        self.italic_off = "\x1b[23m" if mode != "none" else ""
        self.reverse = "\x1b[7m" if mode != "none" else ""
        self.reverse_off = "\x1b[27m" if mode != "none" else ""
        self.underline = "\x1b[4m" if mode != "none" else ""
        self.underline_off = "\x1b[24m" if mode != "none" else ""

        if mode == "true":
            self.purple = lambda v: f"\x1b[38;2;179;136;255m{v}{self.reset}"
            self.purple_bright = lambda v: f"\x1b[38;2;199;166;255m{v}{self.reset}"
            self.purple_dim = lambda v: f"\x1b[38;2;120;80;180m{v}{self.reset}"
            self.purple_dark = lambda v: f"\x1b[38;2;80;40;140m{v}{self.reset}"
            self.green = lambda v: f"\x1b[38;2;134;238;170m{v}{self.reset}"
            self.green_dim = lambda v: f"\x1b[38;2;80;180;120m{v}{self.reset}"
            self.red = lambda v: f"\x1b[38;2;255;128;128m{v}{self.reset}"
            self.red_dim = lambda v: f"\x1b[38;2;180;60;60m{v}{self.reset}"
            self.gold = lambda v: f"\x1b[38;2;255;200;80m{v}{self.reset}"
            self.gold_dim = lambda v: f"\x1b[38;2;180;140;40m{v}{self.reset}"
            self.gray = lambda v: f"\x1b[38;2;190;190;200m{v}{self.reset}"
            self.gray_dim = lambda v: f"\x1b[38;2;120;120;140m{v}{self.reset}"
            self.white = lambda v: f"\x1b[38;2;255;255;255m{v}{self.reset}"
            self.bg_purple = lambda v: f"\x1b[48;2;179;136;255m{v}{self.reset}"
            self.bg_purple_dark = lambda v: f"\x1b[48;2;60;30;100m{v}{self.reset}"
            self.bg_green = lambda v: f"\x1b[48;2;134;238;170m{v}{self.reset}"
            self.bg_gray = lambda v: f"\x1b[48;2;40;40;50m{v}{self.reset}"
            self.bg_dark = lambda v: f"\x1b[48;2;20;20;30m{v}{self.reset}"
        else:
            self.purple = self._make(141)
            self.purple_bright = self._make(147)
            self.purple_dim = self._make(97)
            self.purple_dark = self._make(54)
            self.green = self._make(114)
            self.green_dim = self._make(72)
            self.red = self._make(203)
            self.red_dim = self._make(124)
            self.gold = self._make(214)
            self.gold_dim = self._make(172)
            self.gray = self._make(246)
            self.gray_dim = self._make(240)
            self.white = self._make(15)
            self.bg_purple = self._make_bg(141)
            self.bg_purple_dark = self._make_bg(54)
            self.bg_green = self._make_bg(114)
            self.bg_gray = self._make_bg(237)
            self.bg_dark = self._make_bg(234)

    def _make(self, code256):
        if self.mode == "none":
            return lambda value: value
        prefix = f"\x1b[38;5;{code256}m"
        return lambda value: f"{prefix}{value}{self.reset}"

    def _make_bg(self, code256):
        if self.mode == "none":
            return lambda value: value
        prefix = f"\x1b[48;5;{code256}m"
        return lambda value: f"{prefix}{value}{self.reset}"


def detect_color_mode():
    colorterm = os.environ.get("COLORTERM", "").lower()
    if colorterm in ("truecolor", "24bit"):
        return "true"
    term = os.environ.get("TERM", "").lower()
    if "256color" in term:
        return "256"
    if os.environ.get("WT_SESSION") or "windows terminal" in os.environ.get("TERM_PROGRAM", "").lower():
        return "true"
    if term:
        return "256"
    return "none"


@dataclass(eq=False)
class Message:
    role: str
    text: str
    timestamp: float = 0.0


MAX_MESSAGES = 500


class SplashScreen:
    """Animated splash screen with large StarmentAI logo."""

    LOGO_LINES = [
        "    ███████╗ ██████╗ ██████╗ ███████╗████████╗    ███████╗██╗   ██╗███╗   ██╗███████╗██████╗ ",
        "    ██╔════╝██╔═══██╗██╔══██╗██╔════╝╚══██╔══╝    ██╔════╝██║   ██║████╗  ██║██╔════╝██╔══██╗",
        "    ███████╗██║   ██║██████╔╝█████╗     ██║       ███████╗██║   ██║██╔██╗ ██║█████╗  ██████╔╝",
        "    ╚════██║██║   ██║██╔══██╗██╔══╝     ██║       ╚════██║██║   ██║██║╚██╗██║██╔══╝  ██╔══██╗",
        "    ███████║╚██████╔╝██║  ██║███████╗   ██║       ███████║╚██████╔╝██║ ╚████║███████╗██║  ██║",
        "    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝   ╚═╝       ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝",
    ]

    TAGLINE = "Trợ lý AI chạy cục bộ • Xử lý ngôn ngữ tự nhiên • Tìm kiếm thời gian thực"

    def __init__(self, palette: Palette, width: int, height: int):
        self.palette = palette
        self.width = width
        self.height = height
        self.frame = 0
        self.max_frames = 60
        self.finished = False

    def update(self):
        self.frame += 1
        if self.frame >= self.max_frames:
            self.finished = True

    def render(self) -> List[str]:
        lines = []
        logo_h = len(self.LOGO_LINES)
        total_h = logo_h + 4
        start_y = max(0, (self.height - total_h) // 2)

        for _ in range(start_y):
            lines.append(" " * self.width)

        progress = min(1.0, self.frame / 30.0)
        easing = 1 - (1 - progress) ** 3

        for i, logo_line in enumerate(self.LOGO_LINES):
            visible_chars = int(len(logo_line) * easing)
            visible = logo_line[:visible_chars]
            if self.palette.mode == "true":
                colored = gradient_text(
                    pad_center(visible, self.width),
                    (179, 136, 255),
                    (130, 80, 220)
                )
            else:
                colored = self.palette.purple(pad_center(visible, self.width))
            lines.append(colored)

        lines.append("")

        tagline_progress = max(0.0, min(1.0, (self.frame - 20) / 20.0))
        if tagline_progress > 0:
            tagline_visible = int(len(self.TAGLINE) * tagline_progress)
            tagline = self.TAGLINE[:tagline_visible]
            if self.palette.mode == "true":
                tagline_colored = gradient_text(
                    pad_center(tagline, self.width),
                    (120, 120, 140),
                    (180, 180, 200)
                )
            else:
                tagline_colored = self.palette.gray_dim(pad_center(tagline, self.width))
            lines.append(tagline_colored)

        lines.append("")

        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner = spinner_frames[self.frame % len(spinner_frames)]
        loading_text = f"  {spinner}  Đang khởi tạo StarmentAI..."
        if self.palette.mode == "true":
            loading_colored = gradient_text(
                pad_center(loading_text, self.width),
                (100, 100, 130),
                (150, 150, 180)
            )
        else:
            loading_colored = self.palette.gray_dim(pad_center(loading_text, self.width))
        lines.append(loading_colored)

        while len(lines) < self.height:
            lines.append(" " * self.width)

        return lines[:self.height]


class TUIApp:
    def __init__(self, model="StarmentAI", search_enabled=True, color_mode="256"):
        self.model = model
        self.search_enabled = search_enabled
        self.messages: List[Message] = []
        self.input = ""
        self.cursor = 0
        self.scroll = 0
        self.width = 80
        self.height = 24
        self.color_mode = color_mode
        self.palette = Palette(color_mode)
        self.status = {
            "ollama": "unknown",
            "search": "on" if search_enabled else "off",
        }
        self.pending_submit = None
        self.pending_clear = False
        self.quit = False
        self.generating = False
        self.stream_text = ""
        self.error = None
        self.generation_done = False
        self._wrap_cache = {}
        self.show_splash = True
        self.splash: Optional[SplashScreen] = None
        self.input_focused = False
        self.last_render_time = 0
        self.animation_frame = 0

    def set_size(self, width, height):
        self.width = max(10, int(width))
        self.height = max(5, int(height))
        if self.splash:
            self.splash.width = self.width
            self.splash.height = self.height

    def add_message(self, role, text):
        self.messages.append(Message(role, text, time.time()))
        if len(self.messages) > MAX_MESSAGES:
            oldest = self.messages.pop(0)
            self._wrap_cache.pop(oldest, None)
        self.scroll = 0
        self.show_splash = False

    def clear(self):
        self.messages = []
        self.scroll = 0
        self._wrap_cache = {}
        self.show_splash = True
        self.splash = SplashScreen(self.palette, self.width, self.height)

    def _insert(self, ch):
        if ch >= " " and ch != "\x7f":
            self.input = self.input[: self.cursor] + ch + self.input[self.cursor :]
            self.cursor += 1

    def handle_key(self, key):
        kind = key[0]
        if kind == "char":
            self._insert(key[1])
        elif kind == "backspace":
            if self.cursor > 0:
                self.input = self.input[: self.cursor - 1] + self.input[self.cursor :]
                self.cursor -= 1
        elif kind == "delete":
            if self.cursor < len(self.input):
                self.input = self.input[: self.cursor] + self.input[self.cursor + 1 :]
        elif kind == "left":
            self.cursor = max(0, self.cursor - 1)
        elif kind == "right":
            self.cursor = min(len(self.input), self.cursor + 1)
        elif kind == "home":
            self.cursor = 0
        elif kind == "end":
            self.cursor = len(self.input)
        elif kind == "up":
            self.scroll += 1
        elif kind == "down":
            self.scroll = max(0, self.scroll - 1)
        elif kind == "pageup":
            self.scroll += max(1, self._layout()[1] - 1)
        elif kind == "pagedown":
            self.scroll = max(0, self.scroll - max(1, self._layout()[1] - 1))
        elif kind == "enter":
            self.pending_submit = self.submit()

    def submit(self):
        text = self.input.strip()
        self.input = ""
        self.cursor = 0
        if not text:
            return None
        low = text.lower()
        if low in ("/exit", "/quit"):
            self.quit = True
            return None
        if low == "/clear":
            self.clear()
            self.pending_clear = True
            return None
        if low == "/help":
            self.add_message("system", HELP_TEXT)
            return None
        if low.startswith("/"):
            self.add_message(
                "system", "L\u1ec7nh kh\u00f4ng h\u1ee3p l\u1ec7: " + text + "\nG\u0159 /help \u0111\u1ec3 xem h\u01b0\u1edbng d\u1eabn."
            )
            return None
        return text

    def _layout(self):
        width = self.width
        height = self.height
        header_h = 3
        input_h = 4
        status_h = 1
        chat_h = height - header_h - input_h - status_h
        if chat_h < 3:
            chat_h = 3
            header_h = 2
            input_h = 3
        return header_h, chat_h, input_h, status_h

    def _color_for(self, style):
        p = self.palette
        if style == "user":
            return p.green
        if style == "ai":
            return p.purple
        if style == "system":
            return p.red
        if style == "name_user":
            return lambda value: p.bold + p.green(value) + p.bold_off
        if style == "name_ai":
            return lambda value: p.bold + p.purple(value) + p.bold_off
        if style == "name_system":
            return lambda value: p.bold + p.red(value) + p.bold_off
        if style == "user_bubble":
            return p.bg_green
        if style == "ai_bubble":
            return p.bg_purple_dark
        if style == "system_bubble":
            return p.bg_gray
        return lambda value: value

    def _wrap(self, message, inner):
        cached = self._wrap_cache.get(message)
        if cached is not None and cached[0] == inner:
            return cached[1]
        lines = wrap_text(message.text, inner)
        self._wrap_cache[message] = (inner, lines)
        return lines

    def _format_bubble(self, text: str, align: str, style: str, width: int, is_streaming: bool = False) -> str:
        p = self.palette
        bubble_color = self._color_for(style + "_bubble")
        text_color = self._color_for(style)

        if align == "right":
            prefix = " " * 2
            suffix = " " * 2
            content = pad_left(text, width - 4)
        else:
            prefix = " " * 2
            suffix = " " * 2
            content = pad_right(text, width - 4)

        if is_streaming:
            cursor = "▌"
            if self.animation_frame % 20 < 10:
                cursor = " "
            content = content.rstrip() + p.purple_bright(cursor)

        if style == "user":
            left_cap = p.green("╭─")
            right_cap = p.green("─╮")
            mid_left = p.green("│ ")
            mid_right = p.green(" │")
            bot_left = p.green("╰─")
            bot_right = p.green("─╯")
        elif style == "ai":
            left_cap = p.purple("╭─")
            right_cap = p.purple("─╮")
            mid_left = p.purple("│ ")
            mid_right = p.purple(" │")
            bot_left = p.purple("╰─")
            bot_right = p.purple("─╯")
        else:
            left_cap = p.gray_dim("┌─")
            right_cap = p.gray_dim("─┐")
            mid_left = p.gray_dim("│ ")
            mid_right = p.gray_dim(" │")
            bot_left = p.gray_dim("└─")
            bot_right = p.gray_dim("─┘")

        return ""

    def _render_header(self, width):
        p = self.palette
        lines = []

        title = "S T A R M E N T A I"
        if p.mode == "true":
            title_line = gradient_text(
                pad_center(title, width),
                (179, 136, 255),
                (130, 80, 220)
            )
        else:
            title_line = p.bold + p.purple(pad_center(title, width)) + p.bold_off
        lines.append(title_line)

        tagline = "Trợ lý AI cục bộ  ·  Xử lý ngôn ngữ tự nhiên  ·  Tìm kiếm thời gian thực"
        if p.mode == "true":
            tagline_line = gradient_text(
                pad_center(truncate_text(tagline, width), width),
                (120, 120, 140),
                (180, 180, 200)
            )
        else:
            tagline_line = p.gray_dim(pad_center(truncate_text(tagline, width), width))
        lines.append(tagline_line)

        separator = p.purple_dim("━" * width)
        lines.append(separator)

        return lines

    def _render_chat(self, width, chat_h):
        if self.show_splash and self.splash:
            splash_lines = self.splash.render()
            chat_area = splash_lines[-chat_h:] if len(splash_lines) >= chat_h else splash_lines
            while len(chat_area) < chat_h:
                chat_area.insert(0, " " * width)
            return chat_area

        # Account for: 2 spaces indent + 1 left pad + 1 right pad + 2 box borders = 6
        inner = max(10, width - 6)
        rows = []
        for message in self.messages:
            if message.role == "user":
                name = "Bạn"
                align = "right"
                name_style = "name_user"
                bubble_style = "user"
            elif message.role == "assistant":
                name = self.model
                align = "left"
                name_style = "name_ai"
                bubble_style = "ai"
            else:
                name = "Hệ thống"
                align = "left"
                name_style = "name_system"
                bubble_style = "system"

            name_color = self._color_for(name_style)
            rows.append((name_color(pad_right(f"  {name}", width)), align, "name"))

            for line in self._wrap(message, inner):
                rows.append((line, align, bubble_style))

            rows.append(("", "left", "sep"))

        if self.generating:
            name = self.model
            name_color = self._color_for("name_ai")
            rows.append((name_color(pad_right(f"  {name}", width)), "left", "name"))

            body = self.stream_text
            if not body:
                body = "Đang suy nghĩ..."
            for line in wrap_text(body, inner):
                rows.append((line, "left", "ai"))

            rows.append(("", "left", "sep"))

        total = len(rows)
        max_scroll = max(0, total - chat_h)
        self.scroll = max(0, min(self.scroll, max_scroll))
        end = total - self.scroll
        start = max(0, end - chat_h)
        window = rows[start:end]

        out = []
        for text, align, style in window:
            if style == "name":
                out.append(text)
            elif style == "sep":
                out.append(" " * width)
            else:
                p = self.palette
                bubble_color = self._color_for(style + "_bubble")
                text_color = self._color_for(style)

                if align == "right":
                    content = pad_left(text, inner)
                    line = " " * 2 + bubble_color(" " + content + " ")
                else:
                    content = pad_right(text, inner)
                    line = " " * 2 + bubble_color(" " + content + " ")
                out.append(pad_right(line, width))

        while len(out) < chat_h:
            out.append(" " * width)

        return out

    def _input_view(self, inner):
        value = self.input
        # Calculate visual width and cursor visual position
        visual_width = text_width(value)
        cursor_visual = text_width(value[:self.cursor])
        
        if visual_width <= inner:
            return value, cursor_visual
        # Need to scroll the view
        # Start from cursor position, keep cursor visible
        start_visual = max(0, cursor_visual - inner + 1)
        if start_visual + inner > visual_width:
            start_visual = visual_width - inner
        
        # Find character index corresponding to start_visual
        start_char = 0
        current_visual = 0
        for i, ch in enumerate(value):
            if current_visual >= start_visual:
                break
            current_visual += char_width(ch)
            start_char = i + 1
        
        # Find end character index
        end_visual = start_visual + inner
        end_char = start_char
        current_visual = start_visual
        for i in range(start_char, len(value)):
            cw = char_width(value[i])
            if current_visual + cw > end_visual:
                break
            current_visual += cw
            end_char = i + 1
        
        view = value[start_char:end_char]
        cursor_in_view = cursor_visual - start_visual
        return view, cursor_in_view

    def _render_input(self, width):
        p = self.palette
        inner = max(10, width - 6)
        view, cursor_visual = self._input_view(inner)
        
        # Build buffer with proper cursor handling for wide characters
        buffer = []
        visual_pos = 0
        for ch in view:
            cw = char_width(ch)
            is_cursor = (visual_pos == cursor_visual)
            visual_pos += cw
            
            if is_cursor:
                blink = (self.animation_frame // 15) % 2 == 0
                if blink or self.input_focused:
                    buffer.append(p.reverse + ch + p.reverse_off)
                else:
                    buffer.append(ch)
            else:
                buffer.append(ch)
        
        # If cursor is at the end (after last character)
        if cursor_visual >= visual_pos:
            blink = (self.animation_frame // 15) % 2 == 0
            if blink or self.input_focused:
                buffer.append(p.reverse + " " + p.reverse_off)
            else:
                buffer.append(" ")

        prompt_char = "▸"
        hint = "" if self.input else p.gray_dim("  Nhập tin nhắn... (/help để xem lệnh)")

        if self.generating:
            spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"][self.animation_frame % 10]
            prompt_char = p.purple_bright(spinner)
            hint = p.purple_dim("  Đang tạo phản hồi...")

        content = " " + p.purple(prompt_char) + " " + "".join(buffer) + hint
        content = pad_right(content, width - 4)

        top = p.purple_dim("╭" + "─" * (width - 2) + "╮")
        body = p.purple_dim("│") + content + p.purple_dim("│")
        bottom = p.purple_dim("╰" + "─" * (width - 2) + "╯")

        return [top, body, bottom]

    def _render_status(self, width):
        p = self.palette

        ollama_status = self.status.get("ollama", "unknown")
        if ollama_status == "connected":
            ollama_icon = p.green("●")
            ollama_text = p.green("Đã kết nối")
        elif ollama_status == "error":
            ollama_icon = p.red("●")
            ollama_text = p.red("Lỗi")
        else:
            ollama_icon = p.gold("●")
            ollama_text = p.gold("Đang kết nối...")

        search_status = self.status.get("search", "off")
        if search_status == "on":
            search_icon = p.green("●")
            search_text = p.green("Bật")
        else:
            search_icon = p.gray_dim("●")
            search_text = p.gray_dim("Tắt")

        left = f"{ollama_icon} Ollama: {ollama_text}"
        right = f"{search_icon} Tìm kiếm: {search_text}  ·  /help /clear /exit"

        left_w = text_width(left)
        right_w = text_width(right)

        if left_w + right_w + 3 > width:
            right = truncate_text(right, max(0, width - left_w - 3))
            right_w = text_width(right)

        gap = max(1, width - left_w - right_w)
        line = left + " " * gap + right
        line = pad_right(truncate_text(line, width), width)

        return p.dim + line + p.dim_off

    def render_grid(self, width=None, height=None):
        if width is not None or height is not None:
            if width is not None:
                self.width = max(10, int(width))
            if height is not None:
                self.height = max(5, int(height))
            if self.splash:
                self.splash.width = self.width
                self.splash.height = self.height

        header_h, chat_h, input_h, status_h = self._layout()
        lines = []

        if self.show_splash and self.splash:
            splash_lines = self.splash.render()
            lines.extend(splash_lines[:self.height])
        else:
            lines.extend(self._render_header(self.width))
            lines.extend(self._render_chat(self.width, chat_h))
            lines.extend(self._render_input(self.width))
            lines.append(self._render_status(self.width))

        while len(lines) < self.height:
            lines.append(" " * self.width)

        return lines[:self.height]

    def render(self):
        self.animation_frame += 1
        lines = self.render_grid()
        out = ["\x1b[H"]
        last = len(lines) - 1
        for index, line in enumerate(lines):
            out.append(line)
            out.append("\x1b[K")
            if index != last:
                out.append("\r\n")
        out.append("\x1b[J")
        return "".join(out)


def enable_ansi():
    if os.name == "nt":
        try:
            os.system("")
            import ctypes

            kernel = ctypes.windll.kernel32
            handle = kernel.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            kernel.GetConsoleMode(handle, ctypes.byref(mode))
            kernel.SetConsoleMode(handle, mode.value | 0x0004 | 0x0008)
            kernel.SetConsoleOutputCP(65001)
            kernel.SetConsoleCP(65001)
        except Exception:
            pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class CONSOLE_FONT_INFOEX(ctypes.Structure):
                _fields_ = [
                    ("cbSize", ctypes.c_uint32),
                    ("nFont", ctypes.c_uint32),
                    ("dwFontSize", ctypes.c_coord),
                    ("FontFamily", ctypes.c_uint32),
                    ("FontWeight", ctypes.c_uint32),
                    ("FaceName", ctypes.c_wchar * 32),
                ]

            kernel = ctypes.windll.kernel32
            handle = kernel.GetStdHandle(-11)
            font = CONSOLE_FONT_INFOEX()
            font.cbSize = ctypes.sizeof(CONSOLE_FONT_INFOEX)
            if kernel.GetCurrentConsoleFontEx(handle, False, ctypes.byref(font)):
                for name in ("Cascadia Mono", "Cascadia Code", "Consolas", "Lucida Console", "JetBrains Mono"):
                    font.FaceName = name
                    font.dwFontSize.X = 0
                    font.dwFontSize.Y = 16
                    if kernel.SetCurrentConsoleFontEx(handle, False, ctypes.byref(font)):
                        break
        except Exception:
            pass


def get_size():
    try:
        size = os.get_terminal_size()
        return max(80, size.columns), max(24, size.lines)
    except Exception:
        return 120, 30


def enter_screen():
    sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H")
    sys.stdout.flush()


def leave_screen():
    sys.stdout.write("\x1b[0m\x1b[?25h\x1b[?1049l")
    sys.stdout.flush()


def read_key():
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        code = msvcrt.getwch()
        return (SPECIAL_KEYS.get(code, "unknown"),)
    if ch in ("\r", "\n"):
        return ("enter",)
    if ch == "\x08":
        return ("backspace",)
    if ch == "\t":
        return ("tab",)
    if ch == "\x1b":
        return ("escape",)
    if ch == "\x03":
        return ("ctrl_c",)
    return ("char", ch)


def write_screen(app):
    sys.stdout.write(app.render())
    sys.stdout.flush()


def worker(app, processor, text):
    try:
        for chunk in processor.process_stream(text):
            app.stream_text += chunk
    except Exception as exc:
        app.error = str(exc)
    finally:
        app.generation_done = True


def start_generation(app, processor, text):
    app.add_message("user", text)
    app.generating = True
    app.stream_text = ""
    app.error = None
    app.generation_done = False
    thread = threading.Thread(target=worker, args=(app, processor, text), daemon=True)
    thread.start()


def finalize_generation(app):
    app.generating = False
    if app.error is not None:
        app.add_message("system", "L\u1ed7i: " + app.error)
        app.status["ollama"] = "error"
    else:
        body = app.stream_text.strip()
        if body:
            app.add_message("assistant", body)
            app.status["ollama"] = "connected"
        else:
            app.add_message("system", "Ph\u1ea3n h\u1ed3i tr\u1ed1ng.")
    app.stream_text = ""


def run_loop(app, processor):
    if not HAS_MSVCRT:
        raise RuntimeError("TUI y\u00eau c\u1ea7u msvcrt (ch\u1ec9 h\u1ed7 tr\u1ee3 tr\u00ean Windows).")

    width, height = get_size()
    app.set_size(width, height)
    app.splash = SplashScreen(app.palette, width, height)
    enter_screen()

    last_render = 0.0
    splash_duration = 2.5
    splash_start = time.perf_counter()

    try:
        write_screen(app)
        while not app.quit:
            now = time.perf_counter()

            if app.show_splash and app.splash:
                elapsed = now - splash_start
                if elapsed >= splash_duration:
                    app.show_splash = False
                    app.splash = None
                else:
                    app.splash.update()
                    if now - last_render > 0.05:
                        write_screen(app)
                        last_render = now
                    time.sleep(0.01)
                    continue

            if app.generating:
                if app.generation_done:
                    finalize_generation(app)
                    write_screen(app)
                    last_render = now
                    continue
                if msvcrt.kbhit():
                    key = read_key()
                    if key == ("ctrl_c",):
                        app.quit = True
                if app.stream_text and now - last_render > 0.05:
                    write_screen(app)
                    last_render = now
                else:
                    time.sleep(0.02)
            else:
                if msvcrt.kbhit():
                    key = read_key()
                    if key == ("ctrl_c",):
                        app.quit = True
                    else:
                        app.handle_key(key)
                if app.pending_submit is not None:
                    text = app.pending_submit
                    app.pending_submit = None
                    start_generation(app, processor, text)
                if app.pending_clear:
                    app.pending_clear = False
                    try:
                        processor.clear_conversation()
                    except Exception:
                        pass
                new_width, new_height = get_size()
                if new_width != app.width or new_height != app.height:
                    app.set_size(new_width, new_height)
                    app.splash = SplashScreen(app.palette, new_width, new_height)
                if now - last_render > 0.05:
                    write_screen(app)
                    last_render = now
                else:
                    time.sleep(0.01)
    finally:
        leave_screen()


def main():
    from config import CONFIG
    from core.processor import Processor, ProcessorError

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        sys.stderr.write(
            "StarmentAI TUI y\u00eau c\u1ea7u terminal t\u01b0\u01a1ng t\u00e1c (Windows Terminal/PowerShell).\n"
        )
        return 1

    enable_ansi()
    app = TUIApp(
        model=CONFIG.app_name,
        search_enabled=CONFIG.enable_web_search,
        color_mode=detect_color_mode(),
    )
    try:
        processor = Processor()
    except ProcessorError as exc:
        sys.stderr.write(f"Kh\u00f4ng th\u1ec3 kh\u1edfi t\u1ea1o StarmentAI: {exc}\n")
        return 1
    try:
        run_loop(app, processor)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        leave_screen()
        sys.stderr.write(f"L\u1ed7i TUI: {exc}\n")
        return 1
    finally:
        processor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())