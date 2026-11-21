"""Cached PostgreSQL context with a bounded local fallback during outages."""

from collections import OrderedDict, deque
import logging

logger = logging.getLogger(__name__)


class ConversationContextManager:
    def __init__(self, persistence, settings, memory=None):
        self.persistence = persistence
        self.settings = settings
        self.windows = OrderedDict()
        self.memory = memory

    async def read(self, turn, fallback):
        """Read durable context off the live audio task, with bounded fallbacks."""
        if self.memory is None:
            return fallback
        try:
            turns = await self.memory.window(turn.session_id)
            from backend.app.memory.short_term import memory_turn, turn_order

            cutoff = turn_order(memory_turn(turn))
            return [
                {"mic_id": item["mic_id"], "text": item["text"]}
                for item in turns
                if item["id"] != str(turn.utterance_id) and turn_order(item) < cutoff
            ][-self.settings.context_window_turns :]
        except Exception:
            logger.warning(
                "Memory and PostgreSQL unavailable; using bounded local context",
                exc_info=True,
            )
            return fallback

    def receive(self, turn):
        """Queue the raw utterance immediately, before classification runs."""
        queued = self.persistence.submit(turn=turn)
        window = self.windows.setdefault(
            turn.session_id, deque(maxlen=self.settings.context_window_turns)
        )
        context = list(window)
        window.append({"mic_id": turn.mic_id, "text": turn.raw_text})
        self.windows.move_to_end(turn.session_id)
        while len(self.windows) > self.settings.context_max_sessions:
            self.windows.popitem(last=False)
        return context, queued
