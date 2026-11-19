"""OS-held session lock prevents two local workers from processing one session."""
import hashlib
import os


class SessionLock:
    def __init__(self, db_path, session_id):
        key = hashlib.sha256(session_id.encode()).hexdigest()[:24]
        self.file = open(str(db_path) + "." + key + ".lock", "a+b")
        try:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError("This session already has an active pipeline; close it before resuming") from exc

    def close(self):
        self.file.close()  # Closing releases the lock, including after parent-process death.
