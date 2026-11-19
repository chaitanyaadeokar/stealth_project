"""Offline semantic-label evaluation; recordings without text references get no ASR accuracy claim."""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison",type=Path,default=Path("data/recording_comparison_fixed_auto/comparison.json"))
    parser.add_argument("--output",type=Path,default=Path("data/intent_reconstruction_evaluation.json"))
    args=parser.parse_args()
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ.setdefault("OMP_NUM_THREADS","2")
    import torch
    torch.set_num_threads(2)
    from sentence_classifier import SentenceClassifier
    from profile_phase01 import SAMPLES
    cases=[{"text":text,"expected":label,"source":"fixed_16_item_diagnostic"} for text,label in SAMPLES]
    comparison=json.loads(args.comparison.read_text(encoding="utf-8"))
    for name,result in comparison["models"].items():
        # The fixed 44–68 s passage consists of a question followed by an assertion.
        # These are semantic labels for decoded text, not a human audio transcript.
        if len(result["utterances"]) != 2:
            raise ValueError("Review labels: this evaluator expects the fixed two-turn passage")
        for turn,label in zip(result["utterances"],["question","claim"]):
            cases.append({"text":turn["text"],"expected":label,"source":name+"_recording_decoded_text"})
    classifier=SentenceClassifier(device="cpu")
    started=time.perf_counter()
    classifier.load_model()
    load=time.perf_counter()-started
    rows=[]
    for case in cases:
        started=time.perf_counter()
        result=classifier.classify_intent(case["text"])
        accepted=result["confidence"]>=.5 and not result.get("ambiguous",False)
        rows.append({**case,"predicted":result["top_category"] if accepted else None,
            "classification":result,"wall_sec":time.perf_counter()-started})
        print(json.dumps({"source":case["source"],"expected":case["expected"],"predicted":rows[-1]["predicted"]}),flush=True)
    report={"load_sec":load,"source_sha256":comparison["source_sha256"],"rows":rows,
        "correct":sum(row["predicted"]==row["expected"] for row in rows),
        "abstained":sum(row["predicted"] is None for row in rows),"total":len(rows),
        "note":"Small development diagnostic; not a held-out accuracy benchmark. No ASR reference exists for the discussion."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("correct","abstained","total")}),flush=True)


if __name__=="__main__":
    main()
