"""Compare cached speech models on identical frozen PCM, plus labeled JFK accuracy.

No microphone, network download, external LLM, or mutation of the source session.
The live replay uses a fixed prefix (default 36 s); whole-record decoding uses all
recorded samples. Unlabeled audio gets agreement statistics, never invented WER.
"""
import argparse
import contextlib
import hashlib
import json
import os
import re
import time
from pathlib import Path


def words(text):
    return re.findall(r"\w+", text.lower())


def error_rate(reference, hypothesis):
    ref, hyp = words(reference), words(hypothesis)
    previous = list(range(len(hyp)+1))
    for i, left in enumerate(ref,1):
        current = [i]
        for j, right in enumerate(hyp,1):
            current.append(min(previous[j]+1,current[j-1]+1,previous[j-1]+(left != right)))
        previous = current
    return previous[-1]/max(1,len(ref))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/recording_comparison"))
    parser.add_argument("--replay-seconds", type=float, default=36)
    parser.add_argument("--replay-start", type=float, default=0)
    parser.add_argument("--chunk-seconds", type=float, default=6)
    parser.add_argument("--reuse-whole", type=Path, help="Reuse completed whole-record results only after verifying source hash and settings")
    parser.add_argument("--language", choices=["en","hi","auto"], default="auto")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=1)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS","2")
    os.environ.setdefault("MKL_NUM_THREADS","2")
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from live_whisper_diarize import _setup_cuda_dlls
    _setup_cuda_dlls()
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import get_vad_model
    get_vad_model()
    from audio_capture import AudioSpool
    from capture_runtime import run_capture
    from conversation import ConversationStore
    from llm_pipeline import TranscriptBuffer
    from diarization import DiarizationPipeline
    import logger
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((args.spool/"capture.json").read_text())
    raw = (args.spool/"audio.f32").read_bytes()[:manifest["recorded_samples"]*4]
    audio = np.frombuffer(raw,dtype="<f4").copy()
    fixture = decode_audio("tests/jfk.flac",sampling_rate=16000)
    expected = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"
    frozen = args.output/"frozen-source.f32"
    frozen.write_bytes(raw)
    report = {"source_sha256":hashlib.sha256(raw).hexdigest(),"source_samples":len(audio),
        "duration_sec":len(audio)/16000,"replay_seconds":min(args.replay_seconds,len(audio)/16000),
        "beam_size":args.beam_size,"cpu_threads":4,"num_speakers":args.num_speakers,"models":{}}
    report.update(replay_start=args.replay_start, language=args.language)
    report["chunk_seconds"]=args.chunk_seconds
    reused = json.loads(args.reuse_whole.read_text(encoding="utf-8")) if args.reuse_whole else None
    if reused and (reused["source_sha256"] != report["source_sha256"] or reused["beam_size"] != args.beam_size or reused["cpu_threads"] != 4):
        raise ValueError("Cannot reuse measurements for a different source or decoder configuration")
    if reused:
        report["whole_measurement_note"]=reused.get("measurement_note","Reused prior whole-record measurement")
    logger.configure(log_path=str(args.output/"events.log"))
    diar_root = Path.home()/".cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots"
    diarizer = DiarizationPipeline(str(next(diar_root.iterdir())),device="cpu")
    for name in ("base","large-v3-turbo"):
        root = (Path("data/models/models--Systran--faster-whisper-base/snapshots") if name=="base" else
                Path.home()/".cache/whisper/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots")
        started = time.perf_counter()
        model = WhisperModel(str(next(root.iterdir())),device="cpu",compute_type="int8",cpu_threads=4,local_files_only=True)
        result = {"load_wall_sec":time.perf_counter()-started}
        started=time.perf_counter()
        warm,_=model.transcribe(np.zeros(16000,dtype=np.float32),language="en",beam_size=1,vad_filter=False,condition_on_previous_text=False)
        list(warm)
        result["warmup_wall_sec"]=time.perf_counter()-started
        def decode(samples,language=None):
            started,cpu=time.perf_counter(),time.process_time()
            generator,info=model.transcribe(samples,language=language,beam_size=args.beam_size,
                vad_filter=True,word_timestamps=True,condition_on_previous_text=False)
            decoded=[]
            for segment in generator:
                decoded.append(segment)
                print(json.dumps({"model":name,"decoded_through_sec":segment.end,"source_duration":len(samples)/16000}),flush=True)
            text=" ".join(segment.text.strip() for segment in decoded)
            return {"text":text,"language":info.language,"wall_sec":time.perf_counter()-started,
                    "cpu_sec":time.process_time()-cpu,"audio_sec":len(samples)/16000}
        result["jfk"] = decode(fixture,"en")
        result["jfk"]["wer"] = error_rate(expected,result["jfk"]["text"])
        print(json.dumps({"model":name,"fixture_wall_sec":result["jfk"]["wall_sec"],"fixture_wer":result["jfk"]["wer"]}),flush=True)
        result["recording"] = (dict(reused["models"][name]["recording"], measurement_reused_from=str(args.reuse_whole))
                               if reused else decode(audio,None if args.language=="auto" else args.language))
        result["recording"]["rtf"] = result["recording"]["wall_sec"]/report["duration_sec"]
        if args.reference:
            result["recording"]["wer"] = error_rate(args.reference.read_text(encoding="utf-8"),result["recording"]["text"])
        (args.output/f"{name}-transcript.txt").write_text(result["recording"]["text"],encoding="utf-8")
        print(json.dumps({"model":name,"whole_audio_wall_sec":result["recording"]["wall_sec"],"language":result["recording"]["language"]}),flush=True)
        folder = args.output/name
        store = ConversationStore(folder/"replay.sqlite3")
        buffer = TranscriptBuffer(store,store.create_session())
        spool = AudioSpool(folder/"audio",session_id=buffer.session_id,timeline_offset=args.replay_start,
                           max_pending_seconds=max(120,args.replay_seconds+1))
        replay = audio[int(args.replay_start*16000):int((args.replay_start+args.replay_seconds)*16000)]
        spool.append(replay)
        spool.finish()
        with (folder/"replay-output.txt").open("w",encoding="utf-8") as output, contextlib.redirect_stdout(output):
            result["replay"] = run_capture(model,buffer,diarizer=diarizer,
                recovery=spool.directory,language=args.language,max_seconds=args.chunk_seconds,
                speaker_options={"num_speakers":args.num_speakers},beam_size=args.beam_size)
        result["utterances"] = [{key:row[key] for key in ("text","speaker","start","end","id")}
                                 for row in store.jobs(buffer.session_id)]
        result["speaker_ids"] = sorted({row["speaker"] for row in result["utterances"]})
        result["assembly_state"] = store.audio_speaker_state(buffer.session_id,spool.state["id"])
        store.close()
        report["models"][name] = result
        (args.output/"comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({"model":name,"replay_wall_sec":result["replay"]["wall_sec"],
                          "speaker_ids":result["speaker_ids"],"utterances":len(result["utterances"])}),flush=True)
        del model
    report["transcript_disagreement_rate"] = error_rate(report["models"]["large-v3-turbo"]["recording"]["text"],
                                                        report["models"]["base"]["recording"]["text"])
    report["accuracy_note"] = "User recording has no reference; disagreement is not word error rate. JFK has an explicit reference."
    (args.output/"comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"complete":True,"output":str(args.output/"comparison.json")}),flush=True)


if __name__ == "__main__":
    main()
