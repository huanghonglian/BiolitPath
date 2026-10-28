#!/usr/bin/env python3
"""Run the README workflow using the current Python environment.

Examples:
    python run_biolitpath.py -c test --input query
    python run_biolitpath.py -c test --input text --dry-run
    python run_biolitpath.py -c test --from-step graph --source disease:MESH:D018281

Inputs retain the README layout under case/<case>/. Existing stage outputs
may be overwritten by the underlying tools. --from-step explicitly reruns
that stage and all subsequent selected stages; it does not infer completion.
"""

import argparse
import math
from datetime import datetime
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
STAGES = ("download", "ner", "gnormplus", "gnormplus-post", "tmvar",
          "ner-post", "normalize", "graph", "paths")


def run_command(command, cwd, log):
    """Execute a Python script or JAR in a subprocess and capture output."""
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace") as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise RuntimeError(f"Process exited with code {code}")


def build_steps(args):
    case_dir = ROOT / "case" / args.case
    python = sys.executable
    case_args = ["-c", args.case]
    mode = {"query": "-q", "pmid": "-p", "text": "-t"}.get(args.input, "-t")
    commands = [
        ("download", ROOT, [python, "lit_download.py", *case_args, mode]),
        ("ner", ROOT, [python, "multi_ner.py", *case_args] + (["--no_cuda"] if args.no_cuda else [])),
        ("gnormplus", ROOT / "GNormPlus", ["java", f"-Xmx{args.gnormplus_memory}",
         f"-Xms{args.gnormplus_memory}", "-jar", "GNormPlus.jar",
         str(case_dir / "pubtator"), str(case_dir / "gnormplus_tmp"), "setup.txt"]),
        ("gnormplus-post", ROOT / "GNormPlus", [python, "post_processing_gnormplus.py", *case_args]),
        ("tmvar", ROOT / "tmVar3", ["java", f"-Xmx{args.tmvar_memory}",
         f"-Xms{args.tmvar_memory}", "-jar", "tmVar.jar",
         str(case_dir / "gnormplus"), str(case_dir / "tmvar3")]),
        ("ner-post", ROOT / "multi_ner", [python, "post_process_ner.py", *case_args]),
        ("normalize", ROOT, [python, "normalizer.py", *case_args]),
        ("graph", ROOT / "pathfinder", [python, "co-occurrence.py", *case_args,
         "--weight-method", args.weight_method, "--threshold", str(args.threshold)]),
        ("paths", ROOT / "pathfinder", [python, "pathfinder.py", *case_args,
         "-s", args.source or "", "--max_hop", str(args.max_hop),
         "--min_hop", str(args.min_hop), "-k", str(args.path_num)]),
    ]
    if args.target:
        commands[-1][2].extend(["-t", args.target])
    if args.node_types:
        commands[-1][2].extend(["-n", args.node_types])
    if args.display:
        commands[-1][2].append("-d")
    if args.output:
        # Resolve before switching to the pathfinder working directory.
        commands[-1][2].extend(["--output", str(args.output.expanduser().resolve())])
    return commands[STAGES.index(args.from_step):STAGES.index(args.to_step) + 1]


def require_files(directory, pattern):
    if not any(p.is_file() and p.stat().st_size > 0 for p in directory.glob(pattern)):
        raise RuntimeError(f"Missing nonempty input: {directory / pattern}")


def check_inputs(args, stage):
    """Check consumed artifacts before each stage, including resumed runs."""
    case_dir = ROOT / "case" / args.case
    if stage == "download":
        if args.input == "text":
            require_files(case_dir / "pubtator", f"{args.case}*.PubTator")
        else:
            suffix = "term" if args.input == "query" else "pmid"
            require_files(case_dir, f"{args.case}.{suffix}.txt")
    requirements = {
        "ner": [("pubtator", f"{args.case}*.PubTator")],
        "gnormplus": [("pubtator", f"{args.case}*.PubTator")],
        "gnormplus-post": [("gnormplus_tmp", f"{args.case}*.PubTator*")],
        "tmvar": [("gnormplus", f"{args.case}*.PubTator")],
        "ner-post": [("NERoutput", f"{args.case}*.json"),
                     ("gnormplus", f"{args.case}*.PubTator"),
                     ("tmvar3", f"{args.case}*.PubTator.PubTator")],
        "normalize": [("NERoutput_m", f"{args.case}*.json"),
                      ("gnormplus", f"{args.case}*.PubTator"),
                      ("tmvar3", f"{args.case}*.PubTator.PubTator")],
        "graph": [("NENoutput", f"{args.case}*.json")],
        "paths": [] if (case_dir / "pathfinder" / "graph.pkl").is_file()
        else [("pathfinder", "graph.edgelist"), ("pathfinder", "graph.nodelist")],
    }
    for folder, pattern in requirements.get(stage, []):
        require_files(case_dir / folder, pattern)


def preflight(steps):
    for stage, cwd, command in steps:
        if command[0] == "java":
            if not shutil.which("java"):
                raise RuntimeError("Java executable was not found in PATH")
            jar = command[command.index("-jar") + 1]
            require_files(cwd, jar)
            if stage == "gnormplus":
                require_files(cwd, "setup.txt")
        else:
            require_files(cwd, command[1])
        if stage in ("ner", "normalize"):
            model = ROOT / "model" / ("BiolitNER" if stage == "ner" else "BiolitNEN")
            require_files(model, "config.json")
            if not any(model.glob("*.bin")) and not any(model.glob("*.safetensors")):
                raise RuntimeError(f"Model weights were not found in {model}")
        if stage == "normalize" and not (ROOT / "resources").is_dir():
            raise RuntimeError("Missing resources directory")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--case", required=True)
    parser.add_argument("--input", choices=("query", "pmid", "text"),
                        help="README input mode; required when starting at download")
    parser.add_argument("--from-step", choices=STAGES, default="download")
    parser.add_argument("--to-step", choices=STAGES,
                        help="Last stage (default: paths with --source, otherwise graph)")
    parser.add_argument("-s", "--source")
    parser.add_argument("-t", "--target")
    parser.add_argument("-n", "--node-types", "--node_types")
    parser.add_argument("-k", "--path-num", type=int, default=10)
    parser.add_argument("--max-hop", "--max_hop", type=int, default=2)
    parser.add_argument("--min-hop", "--min_hop", type=int, default=1)
    parser.add_argument("-d", "--display", action="store_true")
    parser.add_argument("-o", "--output", type=Path, help="Path retrieval JSON output path (relative to your current directory)")
    parser.add_argument("--gnormplus-memory", default="40G", help="Java initial/max heap (default: 40G)")
    parser.add_argument("--tmvar-memory", default="5G", help="Java initial/max heap (default: 5G)")
    parser.add_argument("--no-cuda", action="store_true", help="Pass --no_cuda to multi_ner.py")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing or creating files")
    parser.add_argument("--weight-method", choices=("fisher", "npmi", "count"), default="fisher",
                        help="Graph edge weight (default: fisher); applies when running graph stage")
    parser.add_argument("--threshold", type=float,
                        help="Fisher: adjP < threshold (0.05); NPMI: > threshold (0); count: >= threshold (1)")
    args = parser.parse_args(argv)
    args.threshold = args.threshold if args.threshold is not None else {"fisher": 0.05, "npmi": 0.0, "count": 1.0}[args.weight_method]
    if not math.isfinite(args.threshold):
        parser.error("--threshold must be finite")
    if args.weight_method == "fisher" and not 0 < args.threshold <= 1:
        parser.error("Fisher threshold must be in (0, 1]")
    if args.weight_method == "npmi" and not 0 <= args.threshold <= 1:
        parser.error("NPMI threshold must be in [0, 1]; only positive weights are retained")
    if args.weight_method == "count" and (args.threshold < 1 or not args.threshold.is_integer()):
        parser.error("Count threshold must be an integer >= 1")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", args.case):
        parser.error("--case must be a directory name using letters, digits, _, - or .")
    args.to_step = args.to_step or ("paths" if args.source else "graph")
    if STAGES.index(args.from_step) > STAGES.index(args.to_step):
        parser.error("--from-step must precede --to-step")
    if args.from_step == "download" and not args.input:
        parser.error("--input is required when starting at download")
    if args.to_step == "paths" and not args.source:
        parser.error("--source is required for paths")
    if (args.target or args.node_types) and args.to_step != "paths":
        parser.error("--target and --node-types require the paths stage")
    if args.output and args.to_step != "paths":
        parser.error("--output requires the paths stage")
    if args.path_num < 1 or not 1 <= args.min_hop <= args.max_hop:
        parser.error("Require -k >= 1 and 1 <= --min-hop <= --max-hop")
    for memory in (args.gnormplus_memory, args.tmvar_memory):
        if not re.fullmatch(r"[1-9][0-9]*[kKmMgG]?", memory):
            parser.error("Java heap sizes must be positive integers with optional K/M/G suffix")
    return args

def main(argv=None):
    args = parse_args(argv)
    steps = build_steps(args)
    if args.dry_run:
        for stage, cwd, command in steps:
            print(f"[{stage}] cwd={cwd}\n  {shlex.join(command)}")
        return 0
    stage = "preflight"
    try:
        preflight(steps)
        check_inputs(args, steps[0][0])
        case_dir = ROOT / "case" / args.case
        logs = case_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for stage, cwd, command in steps:
            check_inputs(args, stage)
            if stage in ("gnormplus", "tmvar"):
                (case_dir / ("gnormplus_tmp" if stage == "gnormplus" else "tmvar3")).mkdir(exist_ok=True)
            log_path = logs / f"{run_id}_{stage}.log"
            print(f"[{stage}] {shlex.join(command)}\nLog: {log_path}", flush=True)
            started = time.monotonic()
            # Stream combined output to both console and a stage-specific log.
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"cwd: {cwd}\ncall: {shlex.join(command)}\n")
                log.flush()
                try:
                    run_command(command, cwd, log)
                except BaseException:
                    import traceback
                    traceback.print_exc(file=log)
                    raise
                log.write(f"\nCompleted; elapsed: {time.monotonic() - started:.1f}s\n")
            print(f"[{stage}] completed in {time.monotonic() - started:.1f}s", flush=True)
        print(f"Pipeline completed: {case_dir}")
        return 0
    except Exception as exc:
        print(f"Pipeline stopped at {stage}: {exc}", file=sys.stderr)
        print(f"After resolving the error, rerun with --from-step {stage}" if stage != "preflight"
              else "Resolve the missing prerequisites before rerunning.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"\nPipeline interrupted at {stage}.", file=sys.stderr)
        return 130

if __name__ == "__main__":
    sys.exit(main())
