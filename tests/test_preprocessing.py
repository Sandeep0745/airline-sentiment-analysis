import math

import pytest

from src.preprocessing import clean_tweet, clean_tweet_plus, mark_negations, mask_user_handles


class TestCleanTweet:
    def test_basic_tweet(self):
        assert clean_tweet("@united Thanks for the GREAT flight!") == "thanks for the great flight!"

    def test_removes_urls_and_mentions(self):
        assert clean_tweet("@JetBlue look http://t.co/abc123 and www.example.com now") == "look and now"

    def test_decodes_html_entities(self):
        assert clean_tweet("on hold &amp; waiting") == "on hold & waiting"

    def test_normalises_curly_apostrophes(self):
        assert clean_tweet("I don’t know") == "i don't know"

    @pytest.mark.parametrize("missing", [None, float("nan"), math.nan])
    def test_missing_values_become_empty(self, missing):
        assert clean_tweet(missing) == ""

    def test_empty_string(self):
        assert clean_tweet("") == ""


class TestCleanTweetPlus:
    def test_url_is_not_read_as_frown(self):
        # Regression test: the ":/" in "http://" used to become "emo_neg"
        cleaned = clean_tweet_plus("@united thanks http://t.co/abc123 great")
        assert cleaned == "thanks great"
        assert "emo_neg" not in clean_tweet_plus("see https://t.co/XyZ")

    def test_no_url_fragments_leak(self):
        cleaned = clean_tweet_plus("delayed again https://t.co/AbC http www.site.com/page")
        for fragment in ["http", "t.co", "www", "/"]:
            assert fragment not in cleaned

    @pytest.mark.parametrize("text, token", [
        ("love it :)", "emo_pos"), ("love it :-)", "emo_pos"), ("ha :D", "emo_pos"),
        ("so good ;)", "emo_pos"), ("<3 this crew", "emo_pos"),
        ("ugh :(", "emo_neg"), ("meh :/", "emo_neg"), ("bad :-(", "emo_neg"),
    ])
    def test_emoticons(self, text, token):
        assert token in clean_tweet_plus(text).split()

    @pytest.mark.parametrize("text", ["re:delay at gate", "meet at 10:30", "less than <30 min", "ratio 3:2"])
    def test_no_false_emoticons(self, text):
        cleaned = clean_tweet_plus(text)
        assert "emo_pos" not in cleaned and "emo_neg" not in cleaned

    def test_emoji_become_separate_tokens(self):
        assert clean_tweet_plus("great\U0001F60D crew") == "great \U0001F60D crew"

    def test_emoji_variation_selector_removed(self):
        assert "️" not in clean_tweet_plus("love ❤️")

    def test_negation_scope_is_marked(self):
        assert clean_tweet_plus("not happy at all") == "not happy_neg at_neg all_neg"

    def test_negation_stops_at_punctuation(self):
        assert clean_tweet_plus("not good. great crew") == "not good_neg . great crew"

    @pytest.mark.parametrize("negation", ["don't", "don’t", "dont", "can't", "cannot", "isn't", "never"])
    def test_contractions_and_curly_apostrophes(self, negation):
        assert clean_tweet_plus(f"I {negation} like it").endswith("like_neg it_neg")

    def test_emoticon_after_negation_is_not_marked(self):
        assert mark_negations("not emo_pos happy") == "not emo_pos happy_neg"

    def test_elongation_shortened(self):
        assert clean_tweet_plus("sooooo late") == "soo late"

    def test_mentions_removed(self):
        assert "@" not in clean_tweet_plus("@AmericanAir @someone hello")

    def test_documented_example(self):
        assert clean_tweet_plus("@united NOT happy :( sooooo late") == "not happy_neg emo_neg soo_neg late"

    @pytest.mark.parametrize("missing", [None, float("nan"), ""])
    def test_missing_and_empty(self, missing):
        assert clean_tweet_plus(missing) == ""


def test_mask_user_handles_keeps_airlines():
    assert mask_user_handles("@united @dhepburn said hi to @JetBlue") == "@united @user said hi to @JetBlue"
