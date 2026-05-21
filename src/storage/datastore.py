"""External datastore integration for saving conversations."""

import json
import ssl
import urllib.request

from src.config import (
    DATASTORE_TIMEOUT,
    DATASTORE_URL,
    DATASTORE_USER_AGENT,
    DATASTORE_WEBSITE_NAME,
    DATASTORE_WEBSITE_URL,
)
from src.utils.logging import log_step


def save_conversation(session_id: str, user_message: str, ai_reply: str) -> None:
    """
    Save a chat conversation to the external datastore API.

    Designed to be run as a background task.

    Args:
        session_id: Conversation session ID
        user_message: User's message
        ai_reply: AI response
    """
    # Construct the JSON string for ConversationDetails as requested
    conversation_details = json.dumps({
        "User Message": user_message,
        "AI Reply": ai_reply
    })

    payload = {
        "AICC_WebsiteName": DATASTORE_WEBSITE_NAME,
        "AICC_Session_Id": session_id,
        "AICC_WebsiteUrl": DATASTORE_WEBSITE_URL,
        "AICC_ConversationDetails": conversation_details
    }

    data = json.dumps(payload).encode("utf-8")

    # Setup the request
    req = urllib.request.Request(
        DATASTORE_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": DATASTORE_USER_AGENT
        }
    )

    # Ignore SSL verification if staging cert is invalid
    context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(req, context=context, timeout=DATASTORE_TIMEOUT) as response:
            status_code = response.getcode()
            response_body = response.read().decode("utf-8")

            log_step("DATASTORE_SUCCESS", status=status_code, session_id=session_id)
            print(f"Datastore save success: {status_code} - {response_body}")
    except Exception as e:
        log_step("DATASTORE_ERROR", error=str(e), session_id=session_id)
        print(f"Datastore save failed: {e}")

