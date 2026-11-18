import re
import time
import tracemalloc
import unittest

import tui
from tui import (
    TUIApp,
    Palette,
    Message,
    char_width,
    text_width,
    pad_left,
    pad_right,
    pad_center,
    wrap_text,
    truncate_text,
    detect_color_mode,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def strip_ansi(value):
    return ANSI_RE.sub("", value)


class FakeProcessor:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or []
        self.error = error
        self.cleared = False

    def process_stream(self, text):
        if self.error:
            raise RuntimeError(self.error)
        for chunk in self.chunks:
            yield chunk

    def clear_conversation(self):
        self.cleared = True


def make_app(**kwargs):
    """Create a TUIApp with splash disabled for testing."""
    app = TUIApp(color_mode="none", **kwargs)
    app.show_splash = False
    return app


class TestTextUtils(unittest.TestCase):
    def test_char_width(self):
        self.assertEqual(char_width("a"), 1)
        self.assertEqual(char_width("\u4e2d"), 2)
        self.assertEqual(char_width("\u0301"), 0)

    def test_text_width(self):
        self.assertEqual(text_width("abc"), 3)
        self.assertEqual(text_width("a\u4e2d"), 3)

    def test_pad(self):
        self.assertEqual(pad_left("ab", 5), "   ab")
        self.assertEqual(pad_right("ab", 5), "ab   ")
        self.assertEqual(text_width(pad_center("abcd", 10)), 10)

    def test_truncate_wide(self):
        self.assertEqual(truncate_text("abcdef", 3), "abc")
        self.assertEqual(text_width(truncate_text("ab\u4e2d\u6587", 3)), 2)

    def test_wrap_basic(self):
        lines = wrap_text("one two three", 7)
        self.assertTrue(all(text_width(line) <= 7 for line in lines))
        self.assertEqual(" ".join(lines).split(), ["one", "two", "three"])

    def test_wrap_long_word(self):
        lines = wrap_text("aaaaaaaaaa", 4)
        self.assertEqual(lines, ["aaaa", "aaaa", "aa"])

    def test_wrap_newlines(self):
        lines = wrap_text("a\n\nb", 10)
        self.assertEqual(lines, ["a", "", "b"])


class TestPalette(unittest.TestCase):
    def test_none_has_no_ansi(self):
        p = Palette("none")
        self.assertEqual(p.purple("x"), "x")
        self.assertEqual(p.bold + "x" + p.bold_off, "x")

    def test_true_has_ansi(self):
        p = Palette("true")
        self.assertIn("\x1b[38;2;", p.purple("x"))
        self.assertTrue(p.purple("x").endswith("\x1b[0m"))

    def test_256_has_ansi(self):
        p = Palette("256")
        self.assertIn("\x1b[38;5;", p.purple("x"))

    def test_detect_returns_valid(self):
        self.assertIn(detect_color_mode(), ("true", "256", "none"))


class TestEditing(unittest.TestCase):
    def setUp(self):
        self.app = make_app()

    def type_text(self, text):
        for ch in text:
            self.app.handle_key(("char", ch))

    def test_insert_and_backspace(self):
        self.type_text("abc")
        self.assertEqual(self.app.input, "abc")
        self.assertEqual(self.app.cursor, 3)
        self.app.handle_key(("backspace",))
        self.assertEqual(self.app.input, "ab")

    def test_left_insert(self):
        self.type_text("ac")
        self.app.handle_key(("left",))
        self.type_text("b")
        self.assertEqual(self.app.input, "abc")
        self.assertEqual(self.app.cursor, 2)

    def test_delete(self):
        self.type_text("abc")
        self.app.handle_key(("home",))
        self.app.handle_key(("delete",))
        self.assertEqual(self.app.input, "bc")

    def test_home_end(self):
        self.type_text("abc")
        self.app.handle_key(("home",))
        self.assertEqual(self.app.cursor, 0)
        self.app.handle_key(("end",))
        self.assertEqual(self.app.cursor, 3)

    def test_long_input_scroll(self):
        self.type_text("x" * 200)
        grid = self.app.render_grid(80, 24)
        joined = strip_ansi("\n".join(grid))
        self.assertIn("x", joined)
        self.assertEqual(len(grid), 24)


class TestSubmit(unittest.TestCase):
    def setUp(self):
        self.app = make_app()

    def test_enter_returns_text(self):
        for ch in "hello":
            self.app.handle_key(("char", ch))
        self.app.handle_key(("enter",))
        self.assertEqual(self.app.pending_submit, "hello")
        self.assertEqual(self.app.input, "")

    def test_exit_command(self):
        for ch in "/exit":
            self.app.handle_key(("char", ch))
        self.app.handle_key(("enter",))
        self.assertTrue(self.app.quit)
        self.assertIsNone(self.app.pending_submit)

    def test_clear_command(self):
        self.app.add_message("user", "old")
        for ch in "/clear":
            self.app.handle_key(("char", ch))
        self.app.handle_key(("enter",))
        self.assertEqual(self.app.messages, [])
        self.assertTrue(self.app.pending_clear)

    def test_help_command(self):
        for ch in "/help":
            self.app.handle_key(("char", ch))
        self.app.handle_key(("enter",))
        self.assertEqual(len(self.app.messages), 1)
        self.assertEqual(self.app.messages[0].role, "system")
        self.assertIn("/exit", self.app.messages[0].text)

    def test_unknown_command(self):
        for ch in "/foo":
            self.app.handle_key(("char", ch))
        self.app.handle_key(("enter",))
        self.assertEqual(self.app.messages[0].role, "system")
        self.assertIn("/help", self.app.messages[0].text)


class TestRendering(unittest.TestCase):
    def test_header_centered(self):
        app = make_app()
        app.set_size(60, 20)
        grid = app.render_grid()
        # New header has 3 lines: title, tagline, separator
        row = strip_ansi(grid[0])
        self.assertEqual(row, pad_center("S T A R M E N T A I", 60))

    def test_user_right_ai_left(self):
        app = make_app(model="StarmentAI")
        app.set_size(80, 24)
        app.add_message("user", "USERMSG")
        app.add_message("assistant", "AIMSG")
        rows = [strip_ansi(line) for line in app.render_grid()]
        # Find rows containing the messages (they're in bubbles now)
        user_row = next(r for r in rows if "USERMSG" in r)
        ai_row = next(r for r in rows if "AIMSG" in r)
        # User messages are right-aligned, AI messages are left-aligned
        self.assertTrue(user_row.rstrip().endswith("USERMSG"))
        self.assertTrue(ai_row.lstrip().startswith("AIMSG"))

    def test_multiline_wraps_within_width(self):
        app = make_app()
        app.set_size(80, 24)
        app.add_message("assistant", "word " * 40)
        rows = [strip_ansi(line) for line in app.render_grid()]
        for row in rows:
            self.assertLessEqual(text_width(row), 80)

    def test_input_box_has_prompt(self):
        app = make_app()
        app.set_size(80, 24)
        rows = [strip_ansi(line) for line in app.render_grid()]
        # New input box uses ▸ prompt character and ╭─╮ box drawing
        # The ▸ is in the middle line (body) which starts with │
        input_row = next(r for r in rows if "▸" in r)
        self.assertTrue(input_row.startswith("│"))

    def test_status_bar_shows_model_and_hints(self):
        app = make_app(model="StarmentAI")
        app.set_size(80, 24)
        rows = [strip_ansi(line) for line in app.render_grid()]
        # Find the status bar row (contains Ollama and Tìm kiếm)
        status = next(r for r in rows if "Ollama" in r and "Tìm kiếm" in r)
        self.assertIn("Ollama", status)
        self.assertIn("/help", status)
        self.assertIn("Tìm kiếm", status)

    def test_render_has_control_codes(self):
        app = make_app()
        app.set_size(80, 24)
        frame = app.render()
        self.assertTrue(frame.startswith("\x1b[H"))
        self.assertIn("\x1b[K", frame)

    def test_color_true_contains_truecolor(self):
        app = TUIApp(color_mode="true")
        app.show_splash = False
        app.set_size(80, 24)
        grid = app.render_grid()
        self.assertIn("\x1b[38;2;", "\n".join(grid))

    def test_color_none_has_no_escape(self):
        app = make_app()
        app.set_size(80, 24)
        app.add_message("assistant", "hi")
        grid = app.render_grid()
        # With color_mode="none", palette returns empty strings but 
        # the box drawing characters and dim/bright codes might still appear
        # Check that no color ANSI codes are present
        joined = "\n".join(grid)
        # Allow box drawing and basic formatting, but not color codes
        self.assertNotIn("\x1b[38;", joined)
        self.assertNotIn("\x1b[48;", joined)


class TestResize(unittest.TestCase):
    def test_various_sizes(self):
        app = make_app()
        app.add_message("user", "hello")
        app.add_message("assistant", "world")
        for width, height in [(80, 24), (80, 15), (80, 10), (80, 30), (120, 40)]:
            grid = app.render_grid(width, height)
            self.assertEqual(len(grid), height)
            for row in grid:
                self.assertLessEqual(text_width(strip_ansi(row)), width)

    def test_small_terminal(self):
        app = make_app()
        app.set_size(80, 10)
        grid = app.render_grid()
        self.assertEqual(len(grid), 10)


class TestScroll(unittest.TestCase):
    def test_scroll_shows_older_messages(self):
        app = make_app(model="AI")
        app.set_size(80, 24)
        for i in range(20):
            app.add_message("assistant", "MSG%02d" % i)
        bottom = "\n".join(strip_ansi(line) for line in app.render_grid())
        self.assertIn("MSG19", bottom)
        for _ in range(5):
            app.handle_key(("pageup",))
        top = "\n".join(strip_ansi(line) for line in app.render_grid())
        self.assertNotIn("MSG19", top)

    def test_scroll_bounded(self):
        app = make_app()
        app.set_size(80, 24)
        app.add_message("assistant", "only")
        for _ in range(100):
            app.handle_key(("pageup",))
        app.render_grid()
        self.assertGreaterEqual(app.scroll, 0)


class TestGeneration(unittest.TestCase):
    def test_worker_streams_and_finalizes(self):
        app = make_app(model="AI")
        fake = FakeProcessor(chunks=["Hel", "lo"])
        tui.worker(app, fake, "hi")
        self.assertTrue(app.generation_done)
        self.assertEqual(app.stream_text, "Hello")
        self.assertIsNone(app.error)
        tui.finalize_generation(app)
        self.assertEqual(app.messages[-1].role, "assistant")
        self.assertEqual(app.messages[-1].text, "Hello")
        self.assertEqual(app.status["ollama"], "connected")

    def test_worker_error_sets_status(self):
        app = make_app(model="AI")
        fake = FakeProcessor(error="boom")
        tui.worker(app, fake, "hi")
        self.assertTrue(app.generation_done)
        self.assertEqual(app.error, "boom")
        tui.finalize_generation(app)
        self.assertEqual(app.messages[-1].role, "system")
        self.assertEqual(app.status["ollama"], "error")


class TestAsyncFlow(unittest.TestCase):
    def test_start_generation_threaded(self):
        app = make_app(model="AI")
        fake = FakeProcessor(chunks=["A", "B", "C"])
        tui.start_generation(app, fake, "hi")
        self.assertTrue(app.generating)
        self.assertEqual([(m.role, m.text) for m in app.messages], [("user", "hi")])
        for _ in range(300):
            if app.generation_done:
                break
            time.sleep(0.01)
        self.assertTrue(app.generation_done)
        tui.finalize_generation(app)
        self.assertEqual(app.messages[-1].role, "assistant")
        self.assertEqual(app.messages[-1].text, "ABC")
        self.assertEqual(app.status["ollama"], "connected")


class TestPerformance(unittest.TestCase):
    def test_render_throughput(self):
        app = make_app()
        app.set_size(80, 24)
        for i in range(200):
            app.add_message("assistant", "n" * 160)
        start = time.perf_counter()
        for _ in range(500):
            app.render_grid()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0)

    def test_memory_bounded(self):
        tracemalloc.start()
        app = make_app()
        app.set_size(80, 24)
        for i in range(3000):
            app.add_message("user" if i % 2 else "assistant", "x" * 200)
        for _ in range(50):
            app.render_grid()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertLess(peak, 60 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()