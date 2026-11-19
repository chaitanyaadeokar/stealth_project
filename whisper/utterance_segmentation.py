"""Conservative sentence/clause boundaries for spoken English, Hindi and Hinglish."""
import re
import uuid


def is_fragment(text):
    """Detect clear unfinished syntax; do not confuse factuality with truth."""
    value = text.strip().rstrip('.!?।…').strip().lower()
    return bool(re.search(r"(?:^|\s)(?:the|a|an|of|to|for|with|and|or|because|that|which|is|are|was|were|does|do|can|will|would|could|should|very|completely|fully|according|their|your|our|my|का|की|के|और|कि|को|से)$", value)
                or re.fullmatch(r"i|we|they|he|she|it|(?:i|we) (?:think|believe|feel|prefer)|मुझे लगता है|mujhe lagta hai", value))


class UtteranceAssembler:
    """Keep the final clause across acoustic chunks; state is committed with audio.

    Periods at decoder window ends are provisional. A question mark, a real pause,
    a new independent sentence, or a speaker change closes a turn. Limits prevent
    an unpunctuated monologue from retaining unlimited context.
    """
    def __init__(self, state=None, max_seconds=24, max_chars=1600):
        self.pending = dict(state) if state else None
        self.max_seconds, self.max_chars = max_seconds, max_chars

    def state(self):
        return dict(self.pending) if self.pending else None

    def _emit(self):
        if self.pending is None:
            return None
        record = dict(self.pending)
        sources = record.pop("source_ids")
        record["utterance_id"] = (sources[0] if len(sources)==1 else
            uuid.uuid5(uuid.NAMESPACE_URL, "utterance:"+":".join(sources)).hex)
        self.pending = None
        return record

    def feed(self, records, *, watermark=None, final=False):
        ready = []
        for original in records:
            record = dict(original, source_ids=[original["utterance_id"]])
            previous = self.pending
            if previous:
                gap = record.get("start",0) - previous.get("end",0)
                independent = (not is_fragment(previous["text"]) and
                    (explicit_intent(record["text"]) is not None or
                     re.match(r"^(?:I|We|They|He|She|The|A|An|This|That|It|भारत|यह)\b", record["text"])))
                terminal = previous["text"].rstrip().endswith(("?","!","।","."))
                short_question = (len(previous["text"].split()) <= 3 and
                                  previous["text"].rstrip().endswith('?') and not is_fragment(previous["text"]))
                closes = (record["speaker"] != previous["speaker"] or gap >= 0.8 or
                          terminal and independent or short_question)
                if closes:
                    ready.append(self._emit())
                else:
                    # Deduplicate only matching boundary words with overlapping timings.
                    prefix, suffix = previous["text"].rstrip('.?!।… '), record["text"].lstrip()
                    if record.get("start",0) < previous.get("end",0) - 0.02:
                        old_words, new_words = prefix.split(), suffix.split()
                        for count in range(min(6,len(old_words),len(new_words)),0,-1):
                            if [w.lower().strip('.,?!') for w in old_words[-count:]] == [w.lower().strip('.,?!') for w in new_words[:count]]:
                                suffix = " ".join(new_words[count:])
                                break
                    previous["text"] = (prefix + " " + suffix).strip()
                    previous["display_text"] = previous["text"]
                    previous["end"] = max(previous.get("end",0),record.get("end",0))
                    previous["source_ids"] += record["source_ids"]
                    record = previous
            self.pending = record
            if (len(record["text"]) >= self.max_chars or record.get("end",0)-record.get("start",0) >= self.max_seconds):
                ready.append(self._emit())
        if self.pending and (final or watermark is not None and watermark-self.pending.get("end",0) >= 0.8):
            ready.append(self._emit())
        return ready

_ABBREVIATION = re.compile(r"\b(?:mr|mrs|ms|dr|prof|sr|jr|vs|etc|e\.g|i\.e|[a-z])\.$", re.I)
_CLAUSE = re.compile(r"[,;]\s*(?=(?:why|how|what|can you|should we|i think|in my opinion|we should|kya|kyun|lekin|लेकिन|क्या|क्यों)\b)", re.I)


def split_utterances(text):
    """Return (text, start_character, end_character); preserve decimals and titles."""
    boundaries = []
    for match in re.finditer(r"[.!?।]+[\"'”’]*", text):
        end = match.end()
        if match.group().startswith("."):
            if end < len(text) and text[end].isdigit():
                continue
            if _ABBREVIATION.search(text[:end]):
                continue
        if end == len(text) or text[end].isspace() or text[end].isupper() or match.group()[0] in "?!।":
            boundaries.append(end)
    boundaries += [m.start() for m in _CLAUSE.finditer(text)]
    boundaries.append(len(text))
    start, result = 0, []
    for end in sorted(set(boundaries)):
        left, right = start, end
        while left < right and (text[left].isspace() or text[left] in ",;"):
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if right > left:
            result.append((text[left:right], left, right))
        start = end
    return result


def explicit_intent(text):
    """High precision speech-act rules; never infer the truth of an assertion."""
    value = text.strip().lower()
    value = re.sub(r"^(?:(?:but|so|well|then|and|however)[, ]+)+", "", value)
    if re.fullmatch(r"(?:yes|no|ok(?:ay)?|thanks?|thank you|hello|hi|hmm|haan|han|ji|हाँ|जी|ठीक है|नमस्ते|धन्यवाद|good (?:morning|afternoon|evening)(?: everyone| all)?|(?:yeah,?\s*)?i(?:'m| am) done)[.!। ]*", value):
        return "other"
    if re.match(r"^(?:search|find|look up|show me|tell me|please explain|explain|बताओ|बताइए|समझाओ)(?=\s|$|[?!।])", value):
        return "question"
    if (re.match(r"^(?:what|why|where|when|who|which|how|is|are|was|were|can|could|do|does|did|will|would|should|kya|kyun|kaise|क्या|क्यों|कैसे|कौन|कब|कहाँ)(?=\s|$|[?!।])", value)
            or re.match(r"^(?:यह|ये|वह|वो)\s+(?:क्या|क्यों|कैसे|कहाँ|कब)", value)
            or value.endswith(("?", "？"))):
        return "question"
    if re.search(r"\b(?:in my opinion|i (?:think|believe|prefer|feel|agree|disagree)|i (?:don't|do not) (?:think|believe)|should|ought to|we must|let.s debate|better than|meri rai|mujhe lagta|mere hisab)\b|मेरे विचार|मुझे लगता|मेरी राय|चाहिए", value):
        return "opinion"
    # A subject followed by a capability predicate is an assertion, even when
    # its object contains words such as "questions" or "answers". NLI otherwise
    # overweights those topical words in the recorded discussion.
    if re.match(r"^(?:i|we|they|he|she|it|ai|artificial intelligence|teachers?)\s+(?:can|cannot|can't)\s+(?:answer|provide|generate|adapt|handle|improve|replace|understand|help|learn|process)\b",value):
        return "claim"
    # Hindi interrogatives often occur inside the sentence, before the copula.
    if re.search(r"(?:^|\s)(?:क्या|क्यों|कैसे|कौन|कब|कहाँ|kya|kyun|kaise)(?=\s|$|[?!।])", value):
        return "question"
    if len(value.split()) >= 4 and re.search(r"\s(?:है|हैं|था|थी|थे)[।.! ]*$", value):
        return "claim"
    return None
