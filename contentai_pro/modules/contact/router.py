"""Contact form API endpoint."""

import logging
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("contentai_pro.contact")


class ContactForm(BaseModel):
    name: str
    email: str
    subject: str
    message: str


# In-memory store (replace with database in production)
_contacts: list[dict] = []

router = APIRouter(prefix="/api", tags=["contact"])


@router.post("/contact")
async def submit_contact(form: ContactForm):
    """Handle contact form submissions."""
    contact = {
        **form.model_dump(),
        "timestamp": datetime.now().isoformat(),
    }
    _contacts.append(contact)
    logger.info("Contact form submitted by %s", form.email)
    return {"status": "success", "message": "Contact form submitted successfully"}
