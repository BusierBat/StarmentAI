"""
StarmentAI - opencode-style terminal UI (Textual TUI).

Giao diện dạng terminal giống opencode:
- Thanh header: tên app, model, dung lượng context, version.
- Vùng chat: message user/assistant phân biệt màu, assistant render markdown.
- Hộp input có viền ở dưới cùng, streaming từng token.
- Thanh footer: gợi ý phím tắt, trạng thái "thinking".
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterator

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Markdown, Static, TextArea

from config import CONFIG
from core.processor import Processor, ProcessorError

FOOTER_HINT = (
    "enter: gửi · ctrl+enter: xuống dòng · "
    "/help: lệnh · /clear: xóa hội thoại · /exit: thoát"
)

COMMANDS = {
    "/help": "Hiển thị các lệnh",
    "/clear": "Xóa lịch sử hội thoại",
    "/exit": "Thoát StarmentAI",
}


class FakeProcessor:
    """Processor giả cho chế độ demo (không cần Ollama)."""

    def __init__(self) -> None:
        self.last_context_usage: dict[str, Any] = {
            "estimated_tokens": 0,
            "message_budget": CONFIG.num_ctx - CONFIG.context_safety_margin,
            "context_limit": CONFIG.num_ctx,
            "reserved_tokens": CONFIG.context_safety_margin,
            "exceeds_message_budget": False,
        }

    def will_think(self, user_input: str) -> bool:
        return False

    def clear_conversation(self) -> None:
        pass

    def process_stream(self, user_input: str) -> Iterator[str]:
        self.last_context_usage["estimated_tokens"] = 42
        reply = (
            f"(demo) bạn nói: ``{user_input}``\n\n"
            "Đây là giao diện dạng **opencode**.\n"
            "- Dòng trên: header bar (model, context)\n"
            "- Giữa: vùng chat, assistant render markdown\n"
            "- Dưới: hộp input có viền + footer gợi ý phím\n\n"
            "Cài Ollama và chạy `python main.py` để dùng model thật."
        )
        for i in range(0, len(reply), 3):
            yield reply[i : i + 3]
            time.sleep(0.01)

    def close(self) -> None:
        pass


class InputBox(TextArea):
    BINDINGS = [
        Binding("ctrl+enter", "insert_newline", show=False),
        Binding("ctrl+j", "insert_newline", show=False),
    ]

    def __init__(
        self,
        on_send: Callable[[str], None],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            placeholder="Nhắn StarmentAI… (Enter: gửi · Ctrl+Enter: xuống dòng)",
            soft_wrap=True,
            **kwargs,
        )
        self.on_send = on_send
        self._history: list[str] = []
        self._hindex = 0
        self._draft = ""

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            self.action_send()
            return
        if event.key in ("up", "down") and "\n" not in self.text:
            if event.key == "up":
                self.action_history_prev()
            else:
                self.action_history_next()
            event.stop()
            return
        await super()._on_key(event)

    def action_send(self) -> None:
        text = self.text.strip()
        if text:
            self._history.append(text)
        self._hindex = len(self._history)
        if self.on_send:
            self.on_send(text)

    def action_insert_newline(self) -> None:
        self.insert("\n")

    def action_history_prev(self) -> None:
        if "\n" in self.text or not self._history:
            self.action_cursor_up()
            return
        if self._hindex == len(self._history):
            self._draft = self.text
        if self._hindex > 0:
            self._hindex -= 1
            self.text = self._history[self._hindex]

    def action_history_next(self) -> None:
        if "\n" in self.text or not self._history:
            self.action_cursor_down()
            return
        if self._hindex < len(self._history) - 1:
            self._hindex += 1
            self.text = self._history[self._hindex]
        else:
            self._hindex = len(self._history)
            self.text = self._draft


class StarmentTUI(App):
    TITLE = "StarmentAI"
    SUB_TITLE = f"{CONFIG.model_name} · v{CONFIG.app_version}"

    CSS = """
    Screen {
        background: $background;
    }

    #topbar {
        height: 1;
        layout: horizontal;
        align: center middle;
        background: $panel;
    }
    #top-title {
        width: auto;
        padding: 0 1;
        text-style: bold;
    }
    #top-status {
        width: 1fr;
        padding: 0 1;
        text-align: right;
        color: $text-muted;
    }

    #chat {
        height: 1fr;
        padding: 0 1;
    }
    .msg {
        margin: 0 0 1 0;
    }
    .msg-label {
        text-style: bold;
    }
    .user-label { color: #7dcfff; }
    .ai-label { color: #bb9af7; }
    .msg-body { margin: 0 0 0 2; }
    .think { color: $text-muted; text-style: italic; }
    .notice { color: #e0af68; }
    .error { color: #f7768e; }
    Markdown { margin: 0 0 0 2; }

    #input-area {
        height: auto;
        margin: 0 1 0 1;
        padding: 0 1;
        border: round $primary;
        layout: horizontal;
        align: center middle;
    }
    #prompt {
        width: auto;
        color: $primary;
        text-style: bold;
    }
    #input {
        width: 1fr;
        height: 1;
    }

    #footer {
        height: 1;
        background: $panel;
        padding: 0 1;
        color: $text-muted;
        layout: horizontal;
        align: center middle;
    }
    #footer.busy { color: #e0af68; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Thoát", show=False),
        Binding("ctrl+q", "quit", "Thoát", show=False),
    ]

    def __init__(self, demo: bool = False) -> None:
        super().__init__()
        self.demo = demo
        self.processor: Processor | FakeProcessor | None = None
        self._busy = False
        self._buffer = ""
        self._think_box: Static | None = None
        self._stream_box: Static | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="topbar"):
            yield Static(Text.assemble(("⬢ ", "bold #7aa2f7"), ("StarmentAI", "bold")), id="top-title")
            yield Static("", id="top-status")
        yield VerticalScroll(id="chat")
        with Vertical(id="input-area"):
            with Horizontal():
                yield Static("❯", id="prompt")
                yield InputBox(on_send=self.send_message, id="input")
        with Horizontal(id="footer"):
            yield Static(FOOTER_HINT, id="footer-text")

    def on_mount(self) -> None:
        self.chat = self.query_one("#chat", VerticalScroll)
        self.input = self.query_one("#input", InputBox)
        self._update_status()
        self.chat.mount(
            Static(
                Text.assemble(
                    ("⬢ StarmentAI ", "bold #7aa2f7"),
                    ("đã sẵn sàng. Gõ ", "dim"),
                    ("/help", "bold #e0af68"),
                    (" để xem lệnh.", "dim"),
                ),
                classes="msg-body notice",
            )
        )
        try:
            self.processor = (
                FakeProcessor() if self.demo else Processor()
            )
        except Exception as exc:  # noqa: BLE001
            self.processor = None
            self._add_error(f"Khởi tạo Processor lỗi: {exc}")
        self._update_status()
        self.input.focus()

    # --------------------------------------------------
    # UI helpers
    # --------------------------------------------------

    def _update_status(self) -> None:
        usage = (
            self.processor.last_context_usage
            if self.processor is not None
            else None
        )
        used = usage.get("estimated_tokens", 0) if usage else 0
        self.query_one("#top-status", Static).update(
            Text.assemble(
                (CONFIG.model_name, "bold #9ece6a"),
                (f" · ctx {used}/{CONFIG.num_ctx}", "dim"),
                (f" · v{CONFIG.app_version}", "dim"),
            )
        )

    def _set_thinking(self, active: bool) -> None:
        footer = self.query_one("#footer", Horizontal)
        footer.set_styles("color: #e0af68;" if active else "")
        self.query_one("#footer-text", Static).update(
            "⋯ đang suy nghĩ…" if active else FOOTER_HINT
        )

    def _add_user(self, text: str) -> None:
        self.chat.mount(
            Vertical(
                Static("❯ you", classes="msg-label user-label"),
                Static(text, classes="msg-body user-body"),
                classes="msg",
            )
        )
        self.chat.scroll_end()

    def _add_notice(self, text: str) -> None:
        self.chat.mount(Static(text, classes="msg-body notice"))
        self.chat.scroll_end()

    def _add_error(self, text: str) -> None:
        self.chat.mount(Static(text, classes="msg-body error"))
        self.chat.scroll_end()

    def _start_assistant(self) -> None:
        self.chat.mount(
            Static(Text.assemble(("● ", "bold #bb9af7"), ("starmentai", "bold #bb9af7")), classes="msg-label ai-label")
        )
        self._think_box = Static("⋯ thinking…", classes="msg-body think")
        self.chat.mount(self._think_box)
        self.chat.scroll_end()

    def _stream_chunk(self, chunk: str) -> None:
        self._buffer += chunk
        if self._think_box is not None:
            self._think_box.remove()
            self._think_box = None
        if self._stream_box is None:
            self._stream_box = Static("", classes="msg-body stream")
            self.chat.mount(self._stream_box)
        self._stream_box.update(Text(self._buffer))
        self.chat.scroll_end()

    def _stream_done(self) -> None:
        final = self._buffer.strip()
        if self._think_box is not None:
            self._think_box.remove()
            self._think_box = None
        if self._stream_box is not None:
            self._stream_box.remove()
            self._stream_box = None
        if final:
            self.chat.mount(Markdown(final, classes="msg-body md"))
        else:
            self.chat.mount(
                Static("(phản hồi rỗng)", classes="msg-body think")
            )
        self.chat.scroll_end()
        self._finish_turn()

    def _stream_error(self, message: str) -> None:
        if self._think_box is not None:
            self._think_box.remove()
            self._think_box = None
        if self._stream_box is not None:
            self._stream_box.remove()
            self._stream_box = None
        self._add_error(f"⚠ {message}")
        self._finish_turn()

    def _finish_turn(self) -> None:
        self._busy = False
        self._buffer = ""
        self._set_thinking(False)
        self._update_status()
        if self.input is not None:
            self.input.disabled = False
            self.input.focus()

    # --------------------------------------------------
    # Input / commands
    # --------------------------------------------------

    def send_message(self, text: str) -> None:
        if self._busy or not text.strip():
            return

        command = text.strip().lower()
        if command.startswith("/"):
            if self._handle_command(command):
                return
            self._add_notice(
                f"Lệnh không biết: {command}. Gõ /help để xem danh sách."
            )
            self.input.clear()
            return

        self._add_user(text)
        self.input.clear()
        self._busy = True
        self.input.disabled = True
        self._buffer = ""
        self._set_thinking(True)
        self._start_assistant()
        self._generate(text)

    def _handle_command(self, command: str) -> bool:
        if command == "/help":
            body = "\n".join(
                f"  {name:<8} {desc}" for name, desc in COMMANDS.items()
            )
            self.chat.mount(
                Static(
                    Text(body),
                    classes="msg-body notice",
                )
            )
            self.chat.scroll_end()
            return True
        if command == "/clear":
            if self.processor is not None:
                try:
                    self.processor.clear_conversation()
                except Exception as exc:  # noqa: BLE001
                    self._add_error(f"Không xóa được hội thoại: {exc}")
                    return True
            self.chat.remove_children()
            self._add_notice("Đã xóa lịch sử hội thoại.")
            return True
        if command == "/exit":
            self.exit()
            return True
        return False

    # --------------------------------------------------
    # Generation worker
    # --------------------------------------------------

    @work(thread=True, exclusive=True)
    def _generate(self, user_input: str) -> None:
        if self.processor is None:
            self.call_from_thread(
                self._stream_error,
                "Processor chưa được khởi tạo (Ollama không khả dụng?).",
            )
            return
        try:
            for chunk in self.processor.process_stream(user_input):
                self.call_from_thread(self._stream_chunk, chunk)
            self.call_from_thread(self._stream_done)
        except ProcessorError as exc:
            self.call_from_thread(self._stream_error, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(
                self._stream_error, f"Lỗi không xác định: {exc}"
            )


def run_tui(demo: bool = False) -> None:
    StarmentTUI(demo=demo).run()
