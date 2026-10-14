import uuid
from datetime import datetime, timedelta, timezone
from utils.logger import logger

GUEST_SESSIONS_TABLE = "guest_sessions"
SESSION_DURATION_HOURS = 24

def _error(error_type, message):
    """Build a consistent error response."""
    return {
        "success": False,
        "error": {
            "type": error_type,
            "message": message,
        },
    }

def _find_guest_by_name(client, name):
    """Find a guest using exact, case-insensitive, trimmed name matching."""
    response = client.rpc(
        "find_guest_by_normalized_name",
        {"guest_name": name},
    ).execute()

    return response.data or []

def register_guest(client, name):
    """Register a guest and create a 24-hour session."""
    if client is None:
        return _error(
            "SUPABASE_CLIENT_MISSING",
            "A valid Supabase client is required.",
        )

    if not isinstance(name, str) or not name.strip():
        return _error(
            "INVALID_GUEST_NAME",
            "A non-empty guest name is required.",
        )
    name = name.strip()
    try:
        # Exact normalized comparison: '%' and '_' are ordinary characters.
        existing = _find_guest_by_name(client, name)

        if existing:
            logger.warning(
                "Guest registration rejected: name already exists"
            )
            return _error(
                "GUEST_NAME_ALREADY_EXISTS",
                "This guest name is already registered.",
            )

        current_time = datetime.now(timezone.utc)
        user_id = str(uuid.uuid4())
        expires_at = current_time + timedelta(
            hours=SESSION_DURATION_HOURS
        )

        session_data = {
            "id": user_id,
            "name": name,
            "created_at": current_time.isoformat(),
            "last_activity_at": current_time.isoformat(),
            "expires_at": expires_at.isoformat(),
        }

        response = (
            client.table(GUEST_SESSIONS_TABLE)
            .insert(session_data)
            .execute()
        )

        if not response.data:
            raise RuntimeError(
                "Supabase did not confirm guest session creation."
            )

        logger.info("Guest session created successfully")

        return {
            "success": True,
            "name": name,
            "user_id": user_id,
            "expires_at": expires_at.isoformat(),
            "message": "Guest session created successfully.",
            "error": None,
        }

    except Exception:
        logger.exception("Guest registration failed")

        # A concurrent registration may have inserted the same name
        # after the initial lookup. Check again before returning an error.
        try:
            existing = _find_guest_by_name(client, name)

            if existing:
                logger.warning(
                    "Guest registration rejected: duplicate name detected"
                )
                return _error(
                    "GUEST_NAME_ALREADY_EXISTS",
                    "This guest name is already registered.",
                )

        except Exception:
            logger.exception(
                "Failed to verify guest name after registration error"
            )

        return _error(
            "GUEST_REGISTRATION_ERROR",
            "Unable to create the guest session.",
        )


def login_guest(client, name, user_id):
    """Validate a guest name and UUID against an active session."""
    if client is None:
        return _error(
            "SUPABASE_CLIENT_MISSING",
            "A valid Supabase client is required.",
        )

    if (
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(user_id, str)
        or not user_id.strip()
    ):
        return _error(
            "INVALID_GUEST_CREDENTIALS",
            "Guest name and UUID are required.",
        )

    name = name.strip()
    user_id = user_id.strip()

    # Validate UUID format before making a database query.
    try:
        parsed_id = uuid.UUID(user_id)
        user_id = str(parsed_id)
    except (ValueError, AttributeError, TypeError):
        return _error(
            "INVALID_GUEST_CREDENTIALS",
            "Guest name or UUID is incorrect.",
        )

    try:
        response = (
            client.table(GUEST_SESSIONS_TABLE)
            .select("id, name, expires_at")
            .eq("id", user_id)
            .execute()
        )

        sessions = response.data or []

        # UUIDs identify a single row. Verify the name using exact,
        # case-insensitive, trimmed comparison in Python.
        session = next(
            (
                item
                for item in sessions
                if isinstance(item.get("name"), str)
                and item["name"].strip().casefold() == name.casefold()
            ),
            None,
        )

        if not session:
            logger.warning(
                "Guest login rejected: invalid credentials"
            )
            return _error(
                "INVALID_GUEST_CREDENTIALS",
                "Guest name or UUID is incorrect.",
            )

        expires_at = session.get("expires_at")

        if not expires_at:
            return _error(
                "INVALID_SESSION_EXPIRATION",
                "Guest session has no expiration time.",
            )

        expiration_time = datetime.fromisoformat(
            expires_at.replace("Z", "+00:00")
        )

        # Treat timestamps without an explicit timezone as invalid
        # rather than comparing naive and timezone-aware datetimes.
        if expiration_time.tzinfo is None:
            return _error(
                "INVALID_SESSION_EXPIRATION",
                "Guest session expiration must include a timezone.",
            )

        if expiration_time <= datetime.now(timezone.utc):
            logger.warning(
                "Guest login rejected: session expired"
            )
            return _error(
                "USER_SESSION_EXPIRED",
                "Guest session has expired.",
            )

        # Update activity only after credentials and expiration validate.
        client.table(GUEST_SESSIONS_TABLE).update(
            {
                "last_activity_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }
        ).eq("id", user_id).execute()

        logger.info("Guest login successful")

        return {
            "success": True,
            "name": session["name"],
            "user_id": user_id,
            "expires_at": expiration_time.isoformat(),
            "message": "Guest login successful.",
            "error": None,
        }

    except Exception:
        logger.exception("Guest login failed")

        return _error(
            "GUEST_LOGIN_ERROR",
            "Unable to process guest login.",
        )
