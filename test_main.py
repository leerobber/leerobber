"""
Test suite for ContentAI Pro (main.py).

Current coverage baseline: 0% — no tests existed before this file.

Known bugs documented below (marked with xfail where relevant):
  BUG-1  Duplicate route definitions — /generate and /health are each
         registered twice.  FastAPI matches the *first* registration, so
         the second copy is unreachable dead code.
  BUG-2  /contact uses `Form(...)` parameters, but `Form` was not imported.
         (Fixed in this PR: added `Form` to the fastapi import.)
  BUG-3  /contact references `contacts` list that is never defined.
  BUG-4  /contact references `datetime` that is never imported.
  BUG-5  The frontend JS sends JSON to /contact, but the endpoint is
         declared with Form parameters (multipart/urlencoded).  These two
         must be aligned when the endpoint is repaired.
  BUG-6  User store (`users` dict) is in-process memory with no
         persistence; credits reset on every server restart.
  BUG-7  Passwords are stored and compared in plain-text.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient wrapping the FastAPI app."""
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def reset_user_credits():
    """Restore the demo user's credit count after every test."""
    saved = main.users["demo@test.com"]["credits"]
    yield
    main.users["demo@test.com"]["credits"] = saved


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client that returns a canned successful response."""
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_content_block = MagicMock()
    mock_content_block.text = "This is the AI-generated article content."
    mock_message.content = [mock_content_block]
    mock_client.messages.create.return_value = mock_message
    return mock_client


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Smoke tests for the health-check endpoint."""

    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_response_is_json(self, client):
        assert "application/json" in client.get("/health").headers["content-type"]

    def test_status_is_live(self, client):
        assert client.get("/health").json()["status"] == "live"

    def test_service_name_present(self, client):
        assert client.get("/health").json()["service"] == "ContentAI Pro"

    def test_version_field_present(self, client):
        # BUG-1: the second /health registration at line ~1169 omits 'version'
        # and 'features'.  This test verifies the first (active) registration.
        data = client.get("/health").json()
        assert "version" in data

    def test_features_field_present(self, client):
        data = client.get("/health").json()
        assert "features" in data
        assert isinstance(data["features"], list)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestHomepageEndpoint:
    """Basic contract tests for the HTML homepage."""

    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_returns_html_content_type(self, client):
        assert "text/html" in client.get("/").headers["content-type"]

    def test_contains_app_name(self, client):
        assert "ContentAI Pro" in client.get("/").text

    def test_contains_generate_endpoint_reference(self, client):
        # The embedded JS must reference the /generate API.
        assert "/generate" in client.get("/").text

    def test_response_is_not_empty(self, client):
        assert len(client.get("/").text) > 500


# ---------------------------------------------------------------------------
# POST /generate — authentication
# ---------------------------------------------------------------------------

class TestGenerateAuthentication:
    """Tests for credential validation inside POST /generate."""

    def test_unknown_email_returns_401(self, client, mock_anthropic):
        with patch("main.client", mock_anthropic):
            r = client.post("/generate", json={
                "email": "nobody@example.com",
                "password": "demo123",
                "topic": "Remote work",
                "keywords": ["productivity"],
            })
        assert r.status_code == 401
        assert "Invalid credentials" in r.json()["detail"]

    def test_wrong_password_returns_401(self, client, mock_anthropic):
        with patch("main.client", mock_anthropic):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "hunter2",
                "topic": "Remote work",
                "keywords": ["productivity"],
            })
        assert r.status_code == 401
        assert "Invalid credentials" in r.json()["detail"]

    def test_correct_credentials_pass_auth(self, client, mock_anthropic):
        with patch("main.client", mock_anthropic):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Remote work",
                "keywords": ["productivity"],
            })
        assert r.status_code != 401

    def test_auth_failure_does_not_consume_credits(self, client, mock_anthropic):
        main.users["demo@test.com"]["credits"] = 5
        with patch("main.client", mock_anthropic):
            client.post("/generate", json={
                "email": "demo@test.com",
                "password": "wrongpassword",
                "topic": "Remote work",
                "keywords": ["productivity"],
            })
        assert main.users["demo@test.com"]["credits"] == 5


# ---------------------------------------------------------------------------
# POST /generate — API key configuration
# ---------------------------------------------------------------------------

class TestGenerateApiKeyConfig:
    """Tests for the Anthropic client availability check."""

    def test_missing_api_key_returns_500(self, client):
        with patch("main.client", None):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Remote work",
                "keywords": ["productivity"],
            })
        assert r.status_code == 500
        assert "API key not configured" in r.json()["detail"]

    def test_api_key_check_happens_before_auth(self, client):
        # When the client is None the endpoint should short-circuit with 500
        # even before checking credentials.
        with patch("main.client", None):
            r = client.post("/generate", json={
                "email": "bad@example.com",
                "password": "bad",
                "topic": "Remote work",
                "keywords": ["productivity"],
            })
        assert r.status_code == 500


# ---------------------------------------------------------------------------
# POST /generate — credit system
# ---------------------------------------------------------------------------

class TestGenerateCreditSystem:
    """Tests for the in-memory credit gate and deduction logic."""

    def test_zero_credits_returns_402(self, client, mock_anthropic):
        main.users["demo@test.com"]["credits"] = 0
        with patch("main.client", mock_anthropic):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Remote work",
                "keywords": ["productivity"],
            })
        assert r.status_code == 402
        assert "No credits remaining" in r.json()["detail"]

    def test_negative_credits_also_blocked(self, client, mock_anthropic):
        # Credits should never legitimately go negative, but if they do the
        # guard (credits <= 0) should still block generation.
        main.users["demo@test.com"]["credits"] = -1
        with patch("main.client", mock_anthropic):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Remote work",
                "keywords": ["productivity"],
            })
        assert r.status_code == 402

    def test_successful_generation_decrements_credits_by_one(self, client, mock_anthropic):
        main.users["demo@test.com"]["credits"] = 7
        with patch("main.client", mock_anthropic):
            client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Remote work",
                "keywords": ["productivity"],
            })
        assert main.users["demo@test.com"]["credits"] == 6

    def test_response_credits_remaining_matches_stored_value(self, client, mock_anthropic):
        main.users["demo@test.com"]["credits"] = 7
        with patch("main.client", mock_anthropic):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Remote work",
                "keywords": ["productivity"],
            })
        assert r.json()["credits_remaining"] == 6
        assert main.users["demo@test.com"]["credits"] == 6

    def test_multiple_requests_drain_credits_sequentially(self, client, mock_anthropic):
        main.users["demo@test.com"]["credits"] = 3
        payload = {
            "email": "demo@test.com",
            "password": "demo123",
            "topic": "Remote work",
            "keywords": ["productivity"],
        }
        with patch("main.client", mock_anthropic):
            for _ in range(3):
                r = client.post("/generate", json=payload)
                assert r.status_code == 200
            # 4th request should be blocked
            r = client.post("/generate", json=payload)
        assert r.status_code == 402


# ---------------------------------------------------------------------------
# POST /generate — successful generation
# ---------------------------------------------------------------------------

class TestGenerateSuccess:
    """Tests for the happy path: valid credentials, credits available, API ok."""

    def test_returns_200(self, client, mock_anthropic):
        with patch("main.client", mock_anthropic):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "The future of AI",
                "keywords": ["machine learning", "automation"],
            })
        assert r.status_code == 200

    def test_response_contains_content_field(self, client, mock_anthropic):
        with patch("main.client", mock_anthropic):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "The future of AI",
                "keywords": ["machine learning"],
            })
        assert "content" in r.json()

    def test_response_contains_credits_remaining_field(self, client, mock_anthropic):
        with patch("main.client", mock_anthropic):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "The future of AI",
                "keywords": ["machine learning"],
            })
        assert "credits_remaining" in r.json()

    def test_content_value_comes_from_anthropic_response(self, client, mock_anthropic):
        with patch("main.client", mock_anthropic):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Test",
                "keywords": ["test"],
            })
        assert r.json()["content"] == "This is the AI-generated article content."

    def test_anthropic_called_with_correct_model(self, client, mock_anthropic):
        with patch("main.client", mock_anthropic):
            client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "AI in healthcare",
                "keywords": ["diagnostics"],
            })
        kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-5-20250929"

    def test_topic_appears_in_prompt_sent_to_anthropic(self, client, mock_anthropic):
        with patch("main.client", mock_anthropic):
            client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "quantum computing breakthroughs",
                "keywords": ["qubits"],
            })
        prompt = mock_anthropic.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "quantum computing breakthroughs" in prompt

    def test_all_keywords_appear_in_prompt(self, client, mock_anthropic):
        keywords = ["alpha", "beta", "gamma"]
        with patch("main.client", mock_anthropic):
            client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Test",
                "keywords": keywords,
            })
        prompt = mock_anthropic.messages.create.call_args.kwargs["messages"][0]["content"]
        for kw in keywords:
            assert kw in prompt

    def test_empty_keywords_list_accepted(self, client, mock_anthropic):
        with patch("main.client", mock_anthropic):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Test",
                "keywords": [],
            })
        assert r.status_code == 200

    def test_credits_remaining_is_integer(self, client, mock_anthropic):
        with patch("main.client", mock_anthropic):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Test",
                "keywords": ["test"],
            })
        assert isinstance(r.json()["credits_remaining"], int)


# ---------------------------------------------------------------------------
# POST /generate — Anthropic API errors
# ---------------------------------------------------------------------------

class TestGenerateApiErrors:
    """Tests for error propagation when the Anthropic API call fails."""

    def test_anthropic_exception_returns_500(self, client):
        broken = MagicMock()
        broken.messages.create.side_effect = Exception("upstream error")
        with patch("main.client", broken):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Test",
                "keywords": ["test"],
            })
        assert r.status_code == 500

    def test_exception_message_included_in_detail(self, client):
        broken = MagicMock()
        broken.messages.create.side_effect = Exception("rate limit exceeded")
        with patch("main.client", broken):
            r = client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Test",
                "keywords": ["test"],
            })
        assert "rate limit exceeded" in r.json()["detail"]

    def test_credits_not_consumed_on_api_error(self, client):
        """
        Credits are deducted AFTER a successful API response (line ~918 of
        main.py), so a raised exception should leave credits untouched.
        This test documents that correct behaviour.
        """
        main.users["demo@test.com"]["credits"] = 5
        broken = MagicMock()
        broken.messages.create.side_effect = Exception("network timeout")
        with patch("main.client", broken):
            client.post("/generate", json={
                "email": "demo@test.com",
                "password": "demo123",
                "topic": "Test",
                "keywords": ["test"],
            })
        assert main.users["demo@test.com"]["credits"] == 5


# ---------------------------------------------------------------------------
# POST /generate — request validation
# ---------------------------------------------------------------------------

class TestGenerateRequestValidation:
    """Tests for Pydantic model validation on ContentRequest."""

    def test_missing_email_returns_422(self, client):
        r = client.post("/generate", json={
            "password": "demo123",
            "topic": "Test",
            "keywords": ["test"],
        })
        assert r.status_code == 422

    def test_missing_password_returns_422(self, client):
        r = client.post("/generate", json={
            "email": "demo@test.com",
            "topic": "Test",
            "keywords": ["test"],
        })
        assert r.status_code == 422

    def test_missing_topic_returns_422(self, client):
        r = client.post("/generate", json={
            "email": "demo@test.com",
            "password": "demo123",
            "keywords": ["test"],
        })
        assert r.status_code == 422

    def test_missing_keywords_returns_422(self, client):
        r = client.post("/generate", json={
            "email": "demo@test.com",
            "password": "demo123",
            "topic": "Test",
        })
        assert r.status_code == 422

    def test_keywords_must_be_list_not_string(self, client):
        r = client.post("/generate", json={
            "email": "demo@test.com",
            "password": "demo123",
            "topic": "Test",
            "keywords": "productivity, flexibility",  # string, not list
        })
        assert r.status_code == 422

    def test_empty_body_returns_422(self, client):
        assert client.post("/generate", json={}).status_code == 422

    def test_malformed_json_returns_422(self, client):
        r = client.post(
            "/generate",
            content=b"not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /contact — known-broken endpoint (xfail tests)
# ---------------------------------------------------------------------------

class TestContactEndpoint:
    """
    Tests for POST /contact.

    The endpoint currently has three bugs (BUG-3, BUG-4, BUG-5) that prevent
    it from working.  These tests are marked xfail to document the expected
    behaviour once the bugs are fixed:

      BUG-3  `contacts` list is never defined — calling the endpoint raises
             NameError at runtime.
      BUG-4  `datetime` is never imported — same NameError at runtime.
      BUG-5  The frontend JS posts JSON, but the endpoint uses Form()
             parameters.  The content-type mismatch must be resolved by
             either converting the endpoint to accept a JSON body (Pydantic
             model) or updating the client to send form-encoded data.
    """

    @pytest.mark.xfail(
        reason="BUG-3 + BUG-4: `contacts` list and `datetime` not defined in main.py",
        strict=False,
    )
    def test_valid_form_submission_returns_200(self, client):
        r = client.post("/contact", data={
            "name": "Jane Doe",
            "email": "jane@example.com",
            "subject": "Pricing enquiry",
            "message": "I would like to know more about the Professional plan.",
        })
        assert r.status_code == 200

    @pytest.mark.xfail(
        reason="BUG-3 + BUG-4: `contacts` list and `datetime` not defined in main.py",
        strict=False,
    )
    def test_valid_form_submission_returns_success_status(self, client):
        r = client.post("/contact", data={
            "name": "Jane Doe",
            "email": "jane@example.com",
            "subject": "Demo",
            "message": "Hello!",
        })
        assert r.json()["status"] == "success"

    @pytest.mark.xfail(
        reason="BUG-5: endpoint uses Form() but JS sends JSON — content-type mismatch",
        strict=False,
    )
    def test_json_body_also_accepted(self, client):
        """
        The frontend JavaScript POSTs JSON to /contact.  After fixing BUG-5
        the endpoint should accept a JSON body (Pydantic model) so the client
        and server agree on the content type.
        """
        r = client.post("/contact", json={
            "name": "Jane Doe",
            "email": "jane@example.com",
            "subject": "Demo",
            "message": "Hello!",
        })
        assert r.status_code == 200

    def test_missing_name_returns_422(self, client):
        # FastAPI rejects missing Form fields at the framework level (before
        # the function body runs), so this works even while BUG-3/BUG-4 exist.
        r = client.post("/contact", data={
            "email": "jane@example.com",
            "subject": "Demo",
            "message": "Hello!",
        })
        assert r.status_code == 422

    def test_missing_email_returns_422(self, client):
        r = client.post("/contact", data={
            "name": "Jane Doe",
            "subject": "Demo",
            "message": "Hello!",
        })
        assert r.status_code == 422
