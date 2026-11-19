"""Small pyannote adapter; no undeclared sibling WhisperX checkout is required."""


class DiarizationTurns(list):
    """List-compatible turns with pyannote centroids for cross-window identity matching."""
    def __init__(self, turns=(), embeddings=None):
        super().__init__(turns)
        self.embeddings = embeddings or {}


class DiarizationPipeline:
    def __init__(self, model_name, token=None, device="cpu"):
        import torch
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained(model_name, token=token)
        if self.pipeline is None:
            raise RuntimeError("Diarization model unavailable; check model access and HF_TOKEN")
        self.pipeline.to(torch.device(device))

    def __call__(self, audio, **kwargs):
        import torch
        options = {key: value for key, value in kwargs.items() if value is not None}
        expected = options.pop("num_speakers", None)
        if expected is not None:
            # The CLI count is a session constraint. A short window can contain
            # just one of those people; forcing two clusters invents identities.
            options.update(min_speakers=1, max_speakers=expected)
        output = self.pipeline({"waveform": torch.from_numpy(audio).float().unsqueeze(0),
                                "sample_rate": 16000}, **options)
        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", output)
        vectors = getattr(output, "speaker_embeddings", None)
        labels = getattr(output, "speaker_diarization", annotation).labels()
        embeddings = ({label: vectors[index].tolist() for index, label in enumerate(labels)}
                      if vectors is not None and len(vectors) == len(labels) else {})
        return DiarizationTurns([(turn.start, turn.end, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True)], embeddings)


def assign_word_speakers(diarize_turns, transcript, fill_nearest=False, max_gap=0.25):
    """Assign by total overlapping duration; unsupported spans remain UNKNOWN."""
    def speaker_for(start, end):
        if start is None or end is None or end < start:
            return "UNKNOWN"
        scores = {}
        for left, right, speaker in diarize_turns:
            overlap = max(0, min(end, right) - max(start, left))
            if overlap:
                scores[speaker] = scores.get(speaker, 0) + overlap
        if scores:
            return max(scores, key=scores.get)
        if fill_nearest and diarize_turns:
            distances = {}
            for left, right, speaker in diarize_turns:
                distances[speaker] = min(distances.get(speaker, float("inf")), max(left - end, start - right, 0))
            ranked = sorted(distances, key=distances.get)
            if distances[ranked[0]] <= max_gap and (len(ranked) == 1 or distances[ranked[1]] - distances[ranked[0]] > 0.01):
                return ranked[0]
        return "UNKNOWN"

    segments = []
    for segment in transcript.get("segments", []):
        item = dict(segment, speaker=speaker_for(segment["start"], segment["end"]))
        if segment.get("words"):
            item["words"] = [dict(word, speaker=speaker_for(word.get("start"), word.get("end")))
                             for word in segment["words"]]
            # A tiny function-word prefix can straddle the diarization boundary.
            # Repair only when the segment and following word agree on the voice;
            # keep complete short interjections and genuine multiword turns.
            from utterance_segmentation import is_fragment
            for index, word in enumerate(item["words"][:-1]):
                following = item["words"][index+1]
                if (word["speaker"] != item["speaker"] and item["speaker"] != "UNKNOWN"
                        and following["speaker"] == item["speaker"]
                        and is_fragment(word.get("word",""))
                        and word.get("start") is not None and word.get("end") is not None
                        and word["end"]-word["start"] <= 0.6
                        and following.get("start") is not None and following["start"]-word["end"] <= 0.2):
                    word["speaker"] = item["speaker"]
        segments.append(item)
    return {**transcript, "segments": segments}


class SpeakerTracker:
    """Relate rolling-window labels through shared time spans, never label-number identity."""
    def __init__(self, num_speakers=None):
        if num_speakers is not None and (not isinstance(num_speakers, int) or num_speakers < 1):
            raise ValueError("num_speakers must be a positive integer")
        self.previous = []
        self.next_id = 0
        self.num_speakers = num_speakers
        self.prototypes = {}
        self.last_mapping = {}

    def state(self):
        return {"previous": list(self.previous), "next_id": self.next_id,
                "num_speakers": self.num_speakers, "prototypes": dict(self.prototypes),
                "last_mapping": dict(self.last_mapping)}

    def restore(self, state):
        if state:
            self.previous, self.next_id = list(state["previous"]), int(state["next_id"])
            self.prototypes = dict(state.get("prototypes", {}))
            self.last_mapping = dict(state.get("last_mapping", {}))
            if self.num_speakers is None:
                self.num_speakers = state.get("num_speakers")
            if self.num_speakers and self.next_id > self.num_speakers:
                # Upgrade old overlap-only state to the requested bounded identity set.
                self.previous = [(a,b,s) for a,b,s in self.previous if int(s.rsplit('_',1)[-1]) < self.num_speakers]
                self.prototypes = {s:v for s,v in self.prototypes.items() if int(s.rsplit('_',1)[-1]) < self.num_speakers}
                self.last_mapping = {k:s for k,s in self.last_mapping.items() if int(s.rsplit('_',1)[-1]) < self.num_speakers}
                self.next_id = self.num_speakers

    def update(self, turns, window_start, embeddings=None):
        import math
        embeddings = embeddings or getattr(turns, "embeddings", {})
        def normalize(vector):
            norm = math.sqrt(sum(float(v)**2 for v in vector))
            return [float(v)/norm for v in vector] if norm and math.isfinite(norm) else None
        vectors = {key: value for key, vector in embeddings.items() if (value := normalize(vector)) is not None}
        overlap = {}
        for left, right, local in turns:
            for old_left, old_right, stable in self.previous:
                duration = max(0, min(right, old_right) - max(left, old_left))
                if duration:
                    overlap[local, stable] = overlap.get((local, stable), 0) + duration
        mapping, used = {}, set()
        for (local, stable), duration in sorted(overlap.items(), key=lambda pair: pair[1], reverse=True):
            if duration >= 0.1 and local not in mapping and stable not in used:
                vector, prototype = vectors.get(local), self.prototypes.get(stable)
                if vector and prototype and len(vector)==len(prototype) and sum(a*b for a,b in zip(vector,prototype)) < 0.35:
                    continue
                mapping[local] = stable
                used.add(stable)
        # Joint assignment avoids order-dependent swaps when both known voices
        # reappear after a gap. Exact-count sessions are normally small panels.
        if self.num_speakers and self.num_speakers <= 6:
            from itertools import permutations
            local_labels = [label for label in dict.fromkeys(s for _,_,s in turns) if label not in mapping and label in vectors]
            slots = [f"SPEAKER_{i:02d}" for i in range(self.num_speakers) if f"SPEAKER_{i:02d}" not in used]
            if local_labels and len(local_labels) <= len(slots):
                def quality(assignment):
                    return sum(sum(a*b for a,b in zip(self.prototypes[slot],vectors[label]))
                               if slot in self.prototypes and len(self.prototypes[slot])==len(vectors[label]) else 0.65
                               for label,slot in zip(local_labels,assignment))
                best = max(permutations(slots,len(local_labels)),key=quality)
                mapping.update(zip(local_labels,best))
                used.update(best)
                self.next_id = max(self.next_id,max(int(slot.rsplit('_',1)[-1])+1 for slot in best))
        for local in dict.fromkeys(speaker for _, _, speaker in turns):
            if local not in mapping:
                candidates = []
                if local in vectors:
                    for stable, prototype in self.prototypes.items():
                        if stable not in used and len(prototype) == len(vectors[local]):
                            candidates.append((sum(a*b for a,b in zip(prototype,vectors[local])), stable))
                candidates.sort(reverse=True)
                if candidates and (candidates[0][0] >= 0.65 or self.num_speakers and self.next_id >= self.num_speakers):
                    stable = candidates[0][1]
                elif self.num_speakers is None or self.next_id < self.num_speakers:
                    stable = f"SPEAKER_{self.next_id:02d}"
                    self.next_id += 1
                else:
                    slots = [f"SPEAKER_{index:02d}" for index in range(self.num_speakers) if f"SPEAKER_{index:02d}" not in used]
                    if not slots:
                        raise ValueError("Diarization returned more simultaneous identities than --num-speakers")
                    stable = self.last_mapping.get(local)
                    if stable not in slots:
                        stable = slots[0]
                    import logger
                    logger.log_event("speaker_identity_uncertain", local_label=local, assigned=stable,
                                     reason="no_overlap_or_embedding_evidence")
                mapping[local] = stable
                used.add(stable)
        for local, stable in mapping.items():
            if local in vectors:
                old = self.prototypes.get(stable)
                similarity = sum(a*b for a,b in zip(old, vectors[local])) if old and len(old)==len(vectors[local]) else None
                # Do not contaminate a known identity with a forced, dissimilar assignment.
                if old is None:
                    self.prototypes[stable] = vectors[local]
                elif similarity is not None and similarity >= 0.65:
                    self.prototypes[stable] = normalize([0.8*a+0.2*b for a,b in zip(old,vectors[local])])
        self.last_mapping = mapping
        self.previous = [(left, right, mapping[local]) for left, right, local in turns if right >= window_start]
        return self.previous
