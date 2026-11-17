"""
Tests for intent classifier module.
"""

import unittest
from core.intent_classifier import (
    IntentType,
    TimeTrigger,
    SearchTrigger,
    IntentAnalysis,
    is_simple_question,
    detect_time_triggers,
    detect_search_triggers,
    classify_intent,
    analyze_intent,
    should_use_time,
    should_search,
)


class TestSimpleQuestions(unittest.TestCase):
    def test_greeting(self):
        self.assertTrue(is_simple_question("Xin chào"))
        self.assertTrue(is_simple_question("Hello"))
        self.assertTrue(is_simple_question("Chào bạn"))

    def test_thanks(self):
        self.assertTrue(is_simple_question("Cảm ơn"))
        self.assertTrue(is_simple_question("Thanks"))

    def test_bye(self):
        self.assertTrue(is_simple_question("Tạm biệt"))
        self.assertTrue(is_simple_question("Bye"))

    def test_math(self):
        self.assertTrue(is_simple_question("2+2"))
        self.assertTrue(is_simple_question("10 * 5"))

    def test_identity(self):
        self.assertTrue(is_simple_question("Bạn là ai?"))

    def test_complex_not_simple(self):
        self.assertFalse(is_simple_question("Thời tiết hôm nay như thế nào?"))


class TestTimeTriggers(unittest.TestCase):
    def test_current_date(self):
        triggers = detect_time_triggers("Hôm nay là ngày bao nhiêu?")
        self.assertIn(TimeTrigger.CURRENT_DATE, triggers)

    def test_current_time(self):
        triggers = detect_time_triggers("Bây giờ là mấy giờ?")
        self.assertIn(TimeTrigger.CURRENT_TIME, triggers)

    def test_timezone(self):
        triggers = detect_time_triggers("Múi giờ Việt Nam là gì?")
        self.assertIn(TimeTrigger.TIMEZONE, triggers)

    def test_weekday(self):
        triggers = detect_time_triggers("Hôm nay là thứ mấy?")
        self.assertIn(TimeTrigger.WEEKDAY, triggers)

    def test_relative_time(self):
        triggers = detect_time_triggers("Ngày mai là ngày bao nhiêu?")
        self.assertIn(TimeTrigger.RELATIVE_TIME, triggers)

    def test_multiple_triggers(self):
        triggers = detect_time_triggers("Hôm nay thứ mấy, mấy giờ rồi?")
        self.assertIn(TimeTrigger.CURRENT_DATE, triggers)
        self.assertIn(TimeTrigger.WEEKDAY, triggers)
        self.assertIn(TimeTrigger.CURRENT_TIME, triggers)


class TestSearchTriggers(unittest.TestCase):
    def test_current_events(self):
        triggers = detect_search_triggers("Tin tức mới nhất về AI")
        self.assertIn(SearchTrigger.CURRENT_EVENTS, triggers)

    def test_weather(self):
        triggers = detect_search_triggers("Thời tiết Hà Nội hôm nay")
        self.assertIn(SearchTrigger.WEATHER, triggers)

    def test_prices(self):
        triggers = detect_search_triggers("Giá iPhone 15 hiện tại")
        self.assertIn(SearchTrigger.PRICES, triggers)

    def test_software_versions(self):
        triggers = detect_search_triggers("Phiên bản Python mới nhất")
        self.assertIn(SearchTrigger.SOFTWARE_VERSIONS, triggers)

    def test_tech_news(self):
        triggers = detect_search_triggers("Công nghệ AI mới được phát hành")
        self.assertIn(SearchTrigger.TECH_NEWS, triggers)

    def test_person_org_event(self):
        triggers = detect_search_triggers("CEO của OpenAI là ai?")
        self.assertIn(SearchTrigger.PERSON_ORG_EVENT, triggers)

    def test_factual_lookup(self):
        triggers = detect_search_triggers("Machine learning là gì?")
        self.assertIn(SearchTrigger.FACTUAL_LOOKUP, triggers)


class TestIntentClassification(unittest.TestCase):
    def test_greeting_intent(self):
        self.assertEqual(classify_intent("Xin chào"), IntentType.GREETING)

    def test_time_related_intent(self):
        self.assertEqual(classify_intent("Hôm nay là thứ mấy?"), IntentType.TIME_RELATED)

    def test_search_needed_intent(self):
        self.assertEqual(classify_intent("Giá vàng hôm nay"), IntentType.SEARCH_NEEDED)

    def test_technical_intent(self):
        self.assertEqual(classify_intent("Viết hàm Python tính fibonacci"), IntentType.TECHNICAL)

    def test_creative_intent(self):
        self.assertEqual(classify_intent("Viết một bài thơ về mùa thu"), IntentType.CREATIVE)

    def test_simple_question_intent(self):
        self.assertEqual(classify_intent("Tại sao bầu trời xanh?"), IntentType.SIMPLE_QUESTION)


class TestIntentAnalysis(unittest.TestCase):
    def test_time_question_analysis(self):
        analysis = analyze_intent("Bây giờ là mấy giờ?")
        self.assertTrue(analysis.needs_time)
        self.assertFalse(analysis.needs_search)
        self.assertIn(TimeTrigger.CURRENT_TIME, analysis.time_triggers)
        self.assertGreater(analysis.confidence, 0.5)

    def test_search_question_analysis(self):
        analysis = analyze_intent("Giá bitcoin hôm nay")
        self.assertTrue(analysis.needs_search)
        self.assertIn(SearchTrigger.PRICES, analysis.search_triggers)
        # "Giá bitcoin hôm nay" matches PRICES, not CURRENT_EVENTS
        # (CURRENT_EVENTS is for news/tin tức)

    def test_combined_time_and_search(self):
        analysis = analyze_intent("Thời tiết Hà Nội hôm nay")
        self.assertTrue(analysis.needs_time)
        self.assertTrue(analysis.needs_search)
        self.assertIn(TimeTrigger.CURRENT_DATE, analysis.time_triggers)
        self.assertIn(SearchTrigger.WEATHER, analysis.search_triggers)

    def test_simple_question_no_time_no_search(self):
        analysis = analyze_intent("2 + 2 = ?")
        self.assertFalse(analysis.needs_time)
        self.assertFalse(analysis.needs_search)
        self.assertEqual(analysis.intent_type, IntentType.SIMPLE_QUESTION)

    def test_should_use_time(self):
        self.assertTrue(should_use_time("Hôm nay là thứ mấy?"))
        self.assertFalse(should_use_time("Xin chào"))

    def test_should_search(self):
        self.assertTrue(should_search("Giá xăng hôm nay"))
        self.assertFalse(should_search("Cảm ơn bạn"))


if __name__ == "__main__":
    unittest.main()