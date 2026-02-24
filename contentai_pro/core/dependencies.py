from fastapi import HTTPException

from contentai_pro.modules.auth.service import auth_service


async def get_authenticated_user(email: str, password: str) -> dict:
    """Dependency to validate user credentials."""
    user = auth_service.authenticate(email, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user


async def require_credits(email: str, password: str) -> dict:
    """Dependency to validate credentials and check remaining credits."""
    user = await get_authenticated_user(email, password)
    if user["credits"] <= 0:
        raise HTTPException(
            status_code=402,
            detail="No credits remaining. Upgrade to continue generating content!",
        )
    return user
