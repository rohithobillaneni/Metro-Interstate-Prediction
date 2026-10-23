import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

from src.offline_data import hour_category


FRONTEND = Path(__file__).resolve().parents[1] / "frontend.py"


def api_response(payload=None, *, content=None, status=200):
    response = requests.Response()
    response.status_code = status
    response.headers["Content-Type"] = "application/json"
    response._content = content if content is not None else json.dumps(payload).encode()
    return response


class FrontendTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("METRO_API_URL", None)

    def app(self):
        application = AppTest.from_file(FRONTEND, default_timeout=10).run()
        self.assertFalse(application.exception, [error.message for error in application.exception])
        return application

    def assert_handled_error(self, application):
        self.assertFalse(application.exception, [error.message for error in application.exception])
        self.assertEqual(len(application.error), 1)
        self.assertFalse(application.success)
        self.assertFalse(application.get("plotly_chart"))

    def test_initial_page_has_no_api_request_or_prediction(self):
        with patch("requests.post") as post:
            application = self.app()
        post.assert_not_called()
        self.assertEqual(application.sidebar.slider[0].value, 15)
        self.assertEqual(application.sidebar.slider[1].value, 50)
        self.assertEqual(application.sidebar.number_input[0].value, 12)
        self.assertFalse(application.error)
        self.assertFalse(application.success)
        self.assertFalse(application.get("plotly_chart"))

    def test_default_click_sends_valid_payload_with_bounded_timeout(self):
        with patch("requests.post", return_value=api_response({"prediction": 1200.6})) as post:
            application = self.app()
            application.button[0].click().run()
        self.assertFalse(application.exception)
        post.assert_called_once_with(
            "http://127.0.0.1:8000/predict",
            json={
                "holiday": "No", "temp": 15, "clouds_all": 50,
                "weather_main": "Clear", "weekday": "Monday", "hour": 12,
                "hour_category": "Afternoon", "month": "January",
            },
            timeout=10,
        )
        self.assertEqual(len(application.success), 1)
        self.assertEqual(len(application.get("plotly_chart")), 1)
        self.assertFalse(application.error)

    def test_changed_hour_derives_category_and_uses_configured_url(self):
        os.environ["METRO_API_URL"] = "http://traffic-backend:8000/predict"
        with patch("requests.post", return_value=api_response({"prediction": 80})) as post:
            application = self.app()
            for hour in (0, 4, 8, 12, 16, 20, 23):
                with self.subTest(hour=hour):
                    calls_before_change = post.call_count
                    application.sidebar.number_input[0].set_value(hour).run()
                    self.assertEqual(post.call_count, calls_before_change)
                    calls_before_click = post.call_count
                    application.button[0].click().run()
                    self.assertFalse(application.exception)
                    self.assertEqual(post.call_count, calls_before_click + 1)
                    self.assertEqual(post.call_args.args, ("http://traffic-backend:8000/predict",))
                    self.assertEqual(post.call_args.kwargs["json"]["hour"], hour)
                    self.assertEqual(post.call_args.kwargs["json"]["hour_category"], hour_category(hour))
                    self.assertEqual(post.call_args.kwargs["timeout"], 10)

    def test_zero_prediction_and_negative_temperature_are_valid(self):
        with patch("requests.post", return_value=api_response({"prediction": 0})) as post:
            application = self.app()
            application.sidebar.slider[0].set_value(-10).run()
            application.button[0].click().run()
        self.assertFalse(application.exception)
        self.assertEqual(post.call_args.kwargs["json"]["temp"], -10)
        self.assertEqual(len(application.success), 1)
        self.assertIn("0", application.success[0].value)
        self.assertFalse(application.error)

    def test_connection_and_timeout_errors_are_handled(self):
        for failure in (requests.ConnectionError("private transport detail"), requests.Timeout("private transport detail")):
            with self.subTest(failure=type(failure).__name__):
                with patch("requests.post", side_effect=failure):
                    application = self.app()
                    application.button[0].click().run()
                self.assert_handled_error(application)
                self.assertNotIn("private transport detail", application.error[0].value)

    def test_http_error_does_not_decode_or_echo_response_body(self):
        for status in (422, 500, 503):
            with self.subTest(status=status):
                with patch("requests.post", return_value=api_response(content=b"private response detail", status=status)):
                    application = self.app()
                    application.button[0].click().run()
                self.assert_handled_error(application)
                self.assertNotIn("private response detail", application.error[0].value)

    def test_malformed_json_response_is_handled(self):
        with patch("requests.post", return_value=api_response(content=b"private malformed JSON")):
            application = self.app()
            application.button[0].click().run()
        self.assert_handled_error(application)
        self.assertNotIn("private malformed JSON", application.error[0].value)

    def test_invalid_prediction_values_and_response_shapes_are_handled(self):
        invalid = (
            None, [], "text", {}, {"prediction": None}, {"prediction": "42"},
            {"prediction": True}, {"prediction": False}, {"prediction": -1},
            {"prediction": float("nan")}, {"prediction": float("inf")},
            {"prediction": float("-inf")}, {"prediction": 10 ** 400},
        )
        for payload in invalid:
            with self.subTest(payload=payload):
                with patch("requests.post", return_value=api_response(payload)):
                    application = self.app()
                    application.button[0].click().run()
                self.assert_handled_error(application)


if __name__ == "__main__":
    unittest.main()
