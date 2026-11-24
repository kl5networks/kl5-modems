"""The AudioSocket call UUIDs that tell a native-1200 call from a Bell one."""

NATIVE_1200_UUID = "6b6c352d-7570-4d73-9e31-3230306e6174"
WINDOW_UUID_PREFIX = NATIVE_1200_UUID.rsplit("-", 1)[0] + "-"

def window_uuid(window_id):
    return "%s%012d" % (WINDOW_UUID_PREFIX, int(window_id))


def window_of_uuid(uuid):
    """The window id a WINDOW_UUID_PREFIX id carries, or None."""
    uuid = str(uuid or "").lower()
    if not uuid.startswith(WINDOW_UUID_PREFIX):
        return None
    tail = uuid[len(WINDOW_UUID_PREFIX):]
    return int(tail) if len(tail) == 12 and tail.isdigit() else None
