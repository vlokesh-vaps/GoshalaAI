import urllib.request
import json
import ssl
from rag_logging import log_step

DATASTORE_URL = "https://vmsstaging.vapssmartecampus.com:40015/api/ISMDashboardFacade/Save_AI_ChatBot_Conversation/"


def save_conversation(session_id: str, user_message: str, ai_reply: str):
    """
    Sends the chat conversation to the external datastore API.
    Designed to be run as a background task.
    """

    # We construct the JSON string for ConversationDetails as requested.
    conversation_details = json.dumps({
        "User Message": user_message,
        "AI Reply": ai_reply
    })

    payload = {
        "AICC_WebsiteName": "Akshaya_Goshala",
        "AICC_Session_Id": session_id,
        "AICC_WebsiteUrl": "https://vapsfoundation.org/about.html",
        "AICC_ConversationDetails": conversation_details
    }

    data = json.dumps(payload).encode("utf-8")

    # Setup the request
    req = urllib.request.Request(
        DATASTORE_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Technoveda-Chatbot/1.0"
        }
    )

    # Ignore SSL verification if staging cert is invalid (optional, but good for some staging environments)
    context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(req, context=context, timeout=10) as response:
            status_code = response.getcode()
            response_body = response.read().decode("utf-8")

            # Log success in the backend logs
            log_step("DATASTORE_SUCCESS", status=status_code, session_id=session_id)
            print(f"Datastore save success: {status_code} - {response_body}")
    except Exception as e:
        # Log failure in the backend logs without crashing the app
        log_step("DATASTORE_ERROR", error=str(e), session_id=session_id)
        print(f"Datastore save failed: {e}")
