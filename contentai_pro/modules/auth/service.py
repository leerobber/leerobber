from contentai_pro.core.config import settings


class AuthService:
    """In-memory user authentication service."""

    def __init__(self):
        self._users: dict[str, dict] = {
            "demo@test.com": {"password": "demo123", "credits": settings.default_credits}
        }

    def authenticate(self, email: str, password: str) -> dict | None:
        user = self._users.get(email)
        if user and user["password"] == password:
            return {"email": email, **user}
        return None

    def get_credits(self, email: str) -> int:
        user = self._users.get(email)
        return user["credits"] if user else 0

    def deduct_credit(self, email: str) -> int:
        user = self._users.get(email)
        if user and user["credits"] > 0:
            user["credits"] -= 1
            return user["credits"]
        return 0


auth_service = AuthService()
