"""``voxpy`` command-line interface."""

from __future__ import annotations

import argparse
import subprocess
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

from . import __version__, player
from .engines import add_engine_embedding, add_engine_sample, set_provider_extension
from .errors import VoxError
from .validation import Severity, validate_file
from .voxfile import VoxFile


def _out(args: argparse.Namespace, vox: VoxFile) -> Path:
    return vox.write(Path(args.output) if getattr(args, "output", None) else None)


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


# ----------------------------------------------------------------------------- commands


def cmd_create(args: argparse.Namespace) -> int:
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"error: {out} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    fields: dict[str, Any] = {"language": args.language, "gender": args.gender}
    if args.age_range:
        fields["age_range"] = list(args.age_range)
    if args.tag:
        fields["tags"] = args.tag
    vox = VoxFile(args.name, args.description, **fields)
    if args.method:
        vox.set_field("provenance", {"method": args.method, **({"license": args.license} if args.license else {})})
    vox.write(out)
    print(f"created {out} (id {vox.manifest['id']})")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    vox = VoxFile.open(args.file)
    if args.json:
        _print_json({
            "path": str(vox.path),
            "declared_version": vox.source_version,
            "manifest": vox.manifest,
            "entries": [{"path": e.path, "size": e.size, "mime_type": e.mime_type} for e in vox.entries],
            "readiness": vox.readiness.state,
        })
        return 0
    m = vox.manifest
    voice = m.get("voice", {}) if isinstance(m.get("voice"), dict) else {}
    print(f"{vox.path}")
    print(f"  name:        {voice.get('name')}")
    print(f"  id:          {m.get('id')}")
    print(f"  version:     {vox.source_version} (on disk) -> {m.get('vox_version')} (in memory)")
    print(f"  created:     {m.get('created')}" + (f"   modified: {m['modified']}" if m.get("modified") else ""))
    for k in ("language", "gender", "age_range", "tags"):
        if voice.get(k) is not None:
            print(f"  {k + ':':<12} {voice[k]}")
    print(f"  description: {voice.get('description')}")
    prov = m.get("provenance")
    if isinstance(prov, dict):
        print(f"  provenance:  method={prov.get('method')} consent={prov.get('consent')} license={prov.get('license')}")
    if vox.embedding_entries:
        print("  embeddings:")
        for e in vox.embedding_entries:
            mark = "" if e.file in vox else "  [MISSING]"
            lang = e.language or "default"
            print(f"    {e.key}: {e.model} [{e.engine or '-'}; {lang}] {e.file}{mark}")
    if vox.manifest.get("reference_audio"):
        print("  reference_audio:")
        for c in vox.manifest["reference_audio"]:
            print(f"    {c.get('file')}: {c.get('transcript', '')[:60]!r}")
    if vox.extensions:
        print(f"  extensions:  {', '.join(vox.extensions)}")
    print(f"  entries ({len(vox)}):")
    for e in vox.entries:
        print(f"    {e.path} ({e.size} bytes, {e.mime_type})")
    print(f"  readiness:   {vox.readiness.state}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    reports = [validate_file(f, engine=args.engine) for f in args.files]
    failed = False
    for r in reports:
        bad = not r.is_valid or (args.strict and r.warnings)
        failed = failed or bool(bad)
    if args.json:
        _print_json([r.to_dict() for r in reports] if len(reports) > 1 else reports[0].to_dict())
        return 1 if failed else 0
    for r in reports:
        status = "VALID" if r.is_valid else "INVALID"
        if r.is_valid and args.strict and r.warnings:
            status = "INVALID (strict: warnings)"
        print(f"{r.target}: {status}  (declared v{r.declared_version}, schema v{r.schema_version}, "
              f"{r.schema_engine}; {len(r.errors)} errors, {len(r.warnings)} warnings)")
        for i in r.issues:
            if i.severity is Severity.INFO and not args.verbose:
                continue
            print(f"  {i}")
    return 1 if failed else 0


def cmd_models(args: argparse.Namespace) -> int:
    vox = VoxFile.open(args.file)
    samples = player.playable_samples(vox)
    data = {
        "supported_models": sorted(set(vox.supported_models)),
        "embeddings": [e.to_dict() | {"key": e.key, "present": e.file in vox} for e in vox.embedding_entries],
        "samples": [s.__dict__ for s in samples],
        "extensions": sorted(vox.extensions),
    }
    if args.json:
        _print_json(data)
        return 0
    print("models:")
    for m in data["supported_models"]:
        print(f"  {m}")
    print("playable samples:")
    if not samples:
        print("  (none)")
    for s in samples:
        state = "" if s.present else "  [MISSING]"
        print(f"  {s.key}: model={s.model} engine={s.engine or '-'} language={s.language or 'default'}{state}")
    if vox.extensions:
        print(f"server-side / extension providers: {', '.join(sorted(vox.extensions))}")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    vox = VoxFile.open(args.file)
    if args.list or not args.model:
        return cmd_models(argparse.Namespace(file=args.file, json=False))
    if args.extract:
        r, out = player.extract(vox, args.model, args.extract, args.language)
        print(f"{r.path} ({r.via}) -> {out} ({len(r.data or b'')} bytes)")
        return 0
    r = player.play(vox, args.model, args.language, player=args.player)
    print(f"played {r.path} ({r.via})")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    vox = VoxFile.open(args.file)
    if args.entry:
        e = vox[args.entry]
        if e is None:
            print(f"error: no entry '{args.entry}'", file=sys.stderr)
            return 1
        out = Path(args.output)
        if out.is_dir():
            out = out / Path(e.path).name
        out.write_bytes(e.data)
        print(f"{e.path} -> {out} ({e.size} bytes)")
        return 0
    if not args.model:
        print("error: give a MODEL query or --entry PATH", file=sys.stderr)
        return 2
    if args.clone_prompt:
        r = vox.resolve_clone_prompt(args.model, args.language)
        if r is None or r.data is None:
            print(f"error: no clone prompt for '{args.model}'", file=sys.stderr)
            return 1
        out = Path(args.output)
        if out.is_dir():
            out = out / Path(r.path).name
        out.write_bytes(r.data)
    else:
        r, out = player.extract(vox, args.model, args.output, args.language)
    print(f"{r.path} ({r.via}) -> {out} ({len(r.data or b'')} bytes)")
    return 0


def cmd_add_embedding(args: argparse.Namespace) -> int:
    vox = VoxFile.open(args.file)
    data = Path(args.source).read_bytes()
    e = add_engine_embedding(vox, data, engine=args.engine, model=args.model, name=args.name or Path(args.source).name,
                             model_slug=args.slug, language=args.language, format=args.format,
                             description=args.description, key=args.key)
    out = _out(args, vox)
    print(f"added {e.file} as embeddings.{e.key} -> {out}")
    return 0


def cmd_add_sample(args: argparse.Namespace) -> int:
    vox = VoxFile.open(args.file)
    e = add_engine_sample(vox, Path(args.source).read_bytes(), engine=args.engine, model=args.model,
                          model_slug=args.slug, language=args.language, description=args.description)
    out = _out(args, vox)
    print(f"added {e.file} as embeddings.{e.key} -> {out}")
    return 0


def cmd_set_extension(args: argparse.Namespace) -> int:
    vox = VoxFile.open(args.file)
    if args.remove:
        if vox.remove_extension(args.provider) is None:
            print(f"error: no extensions.{args.provider}", file=sys.stderr)
            return 1
        out = _out(args, vox)
        print(f"removed extensions.{args.provider} -> {out}")
        return 0
    data: dict[str, Any] = {}
    if args.json_file:
        data.update(json.loads(Path(args.json_file).read_text(encoding="utf-8")))
    if args.json:
        data.update(json.loads(args.json))
    for kv in args.set or []:
        if "=" not in kv:
            print(f"error: --set expects KEY=VALUE, got '{kv}'", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        data[k] = _parse_value(v)
    if not data:
        print("error: nothing to set (use --json, --json-file or --set)", file=sys.stderr)
        return 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        set_provider_extension(vox, args.provider, data, merge=args.merge)
    for w in caught:
        print(f"warning: {w.message}", file=sys.stderr)
    out = _out(args, vox)
    print(f"set extensions.{args.provider} ({', '.join(data)}) -> {out}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    vox = VoxFile.open(args.file)
    missing = [p for p in args.paths if vox.remove(p) is None]
    for p in missing:
        print(f"warning: no entry '{p}'", file=sys.stderr)
    if len(missing) == len(args.paths):
        return 1
    out = _out(args, vox)
    print(f"removed {len(args.paths) - len(missing)} entr{'y' if len(args.paths) - len(missing) == 1 else 'ies'} -> {out}")
    return 0


# ----------------------------------------------------------------------------- gemini


def _gemini_client():
    from .providers.gemini import GeminiVoicesClient

    return GeminiVoicesClient()


def _summary(ext: dict[str, Any] | None) -> str:
    if not ext:
        return "(none)"
    return f"voice_id={ext.get('voice_id')} expires={ext.get('expire_time')}"


def cmd_gemini(args: argparse.Namespace) -> int:
    from .providers import gemini as g

    client = _gemini_client()
    if args.gemini_cmd == "list":
        _print_json(client.list_voices(None if args.type == "all" else args.type))
        return 0
    if args.gemini_cmd == "get":
        _print_json(g.redact_response(client.get_voice(args.voice_id)))
        return 0
    if args.gemini_cmd == "delete":
        if not args.yes:
            print("error: deleting a server voice is irreversible; pass --yes", file=sys.stderr)
            return 2
        vox = VoxFile.open(args.vox) if args.vox else None
        removed = g.delete_voice(client, args.voice_id, vox)
        if vox is not None and removed:
            vox.write()
        print(f"deleted {args.voice_id}" + (f"; removed from {args.vox}: {', '.join(removed)}" if removed else ""))
        return 0
    if args.gemini_cmd == "synthesize":
        ext = g.GeminiAdapter().read_extension(VoxFile.open(args.file))
        if not ext or not ext.get("voice_id"):
            print(f"error: {args.file} has no extensions.google.voice_id", file=sys.stderr)
            return 1
        text = Path(args.text_file).read_text(encoding="utf-8").strip() if args.text_file else args.text
        if not text:
            print("error: pass TEXT or --text-file", file=sys.stderr)
            return 2
        audio = client.synthesize(ext["voice_id"], text, style=args.style)
        Path(args.output).write_bytes(audio)
        print(f"{args.file}: synthesized {len(audio)}B with {ext['voice_id']} -> {args.output}")
        if args.play:
            import subprocess
            from .player import find_player
            subprocess.run([*find_player(), args.output], check=True)
        return 0
    prompt = getattr(args, "prompt", None)
    if getattr(args, "prompt_file", None):
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.gemini_cmd == "create":
        vox = VoxFile.open(args.file)
        voice = g.create_voice_for_vox(vox, client, prompt=prompt, display_name=args.display_name, gender=args.gender,
                                       language_code=args.language, force=args.force)
        vox.write()
        print(f"{args.file}: created {_summary(voice.extension)} sample={sum(len(s.data) for s in voice.samples)}B")
        return 0
    if args.gemini_cmd == "refresh":
        rc = 0
        for f in args.files:
            vox = VoxFile.open(f)
            try:
                voice = g.refresh_vox(vox, client)
            except VoxError as e:
                print(f"{f}: error: {e}", file=sys.stderr)
                rc = 1
                continue
            vox.write()
            print(f"{f}: refreshed {_summary(voice.extension)} sample={sum(len(s.data) for s in voice.samples)}B")
        return rc
    if args.gemini_cmd == "update":
        vox = VoxFile.open(args.file)
        res = g.update_vox(vox, client, prompt=prompt, display_name=args.display_name, gender=args.gender,
                           language_code=args.language)
        msg = f"{args.file}: replaced {res.old_voice_id} -> {res.new_voice_id}"
        if not res.old_deleted:
            msg += f" (old voice NOT deleted: {res.delete_error})"
        print(msg)
        return 0 if res.old_deleted else 1
    return 2


# ----------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voxpy", description="Read, write, validate and play .vox voice identity files.")
    p.add_argument("--version", action="version", version=f"voxpy {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def out_opt(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-o", "--output", help="write to this path instead of updating FILE in place")

    sp = sub.add_parser("create", help="create a new .vox")
    sp.add_argument("output", help="destination .vox")
    sp.add_argument("--name", required=True)
    sp.add_argument("--description", required=True)
    sp.add_argument("--language")
    sp.add_argument("--gender", choices=["male", "female", "nonbinary", "neutral"])
    sp.add_argument("--age-range", nargs=2, type=int, metavar=("MIN", "MAX"))
    sp.add_argument("--tag", action="append")
    sp.add_argument("--method", choices=["designed", "synthesized", "cloned", "preset", "hybrid"])
    sp.add_argument("--license")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_create)

    sp = sub.add_parser("inspect", help="show manifest and entries")
    sp.add_argument("file")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("validate", help="validate .vox archives or manifest.json files")
    sp.add_argument("files", nargs="+")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--strict", action="store_true", help="treat warnings as failures")
    sp.add_argument("-v", "--verbose", action="store_true", help="also show info-level notes")
    sp.add_argument("--engine", choices=["auto", "builtin", "jsonschema"], default="auto",
                    help="schema engine (auto = jsonschema if installed)")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("models", help="list models, playable samples and providers")
    sp.add_argument("file")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("play", help="play the sample for a TTS model (e.g. 0.6b, 1.7b, gemini)")
    sp.add_argument("file")
    sp.add_argument("model", nargs="?")
    sp.add_argument("-l", "--language", help="BCP 47 tag; falls back exact -> base -> default")
    sp.add_argument("--player", help="player binary (default: afplay, ffplay, aplay)")
    sp.add_argument("--extract", metavar="PATH", help="write the WAV here instead of playing")
    sp.add_argument("--list", action="store_true", help="list playable samples")
    sp.set_defaults(func=cmd_play)

    sp = sub.add_parser("extract", help="extract a model's sample (or clone prompt / any entry)")
    sp.add_argument("file")
    sp.add_argument("model", nargs="?")
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("-l", "--language")
    sp.add_argument("--clone-prompt", action="store_true", help="extract the clone prompt instead of the sample")
    sp.add_argument("--entry", help="extract this archive path verbatim")
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("add-embedding", help="add a local engine binary under embeddings/<engine>/<slug>/")
    sp.add_argument("file")
    sp.add_argument("source", help="local file to add")
    sp.add_argument("--engine", required=True)
    sp.add_argument("--model", required=True, help="fully qualified model id")
    sp.add_argument("--slug", help="model path segment (e.g. 0.6b); default derived from --model")
    sp.add_argument("--name", help="file name in the archive (default: source file name)")
    sp.add_argument("-l", "--language")
    sp.add_argument("--format")
    sp.add_argument("--description")
    sp.add_argument("--key", help="manifest key (default derived from the path)")
    out_opt(sp)
    sp.set_defaults(func=cmd_add_embedding)

    sp = sub.add_parser("add-sample", help="add engine-generated sample audio (language-neutral by default)")
    sp.add_argument("file")
    sp.add_argument("source", help="WAV file")
    sp.add_argument("--engine", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--slug")
    sp.add_argument("-l", "--language")
    sp.add_argument("--description")
    out_opt(sp)
    sp.set_defaults(func=cmd_add_sample)

    sp = sub.add_parser("set-extension", help="set extensions.<provider> (server-side voice handles)")
    sp.add_argument("file")
    sp.add_argument("provider", help="provider id, e.g. elevenlabs, google, openai, apple")
    sp.add_argument("--json", help="JSON object")
    sp.add_argument("--json-file")
    sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="value parsed as JSON when possible")
    sp.add_argument("--merge", action="store_true", help="merge into the existing object")
    sp.add_argument("--remove", action="store_true", help="remove extensions.<provider>")
    out_opt(sp)
    sp.set_defaults(func=cmd_set_extension)

    sp = sub.add_parser("remove", help="remove archive entries (manifest kept in sync)")
    sp.add_argument("file")
    sp.add_argument("paths", nargs="+")
    out_opt(sp)
    sp.set_defaults(func=cmd_remove)

    gp = sub.add_parser("gemini", help="Gemini 3.8 Flash TTS voices (needs GEMINI_API_KEY)")
    gsub = gp.add_subparsers(dest="gemini_cmd", required=True)

    def voice_opts(s: argparse.ArgumentParser) -> None:
        s.add_argument("--prompt", help="voice design prompt (default: voice.description)")
        s.add_argument("--prompt-file")
        s.add_argument("--display-name", help="default: voice.name")
        s.add_argument("--gender", choices=["male", "female"])
        s.add_argument("-l", "--language", help="language_code (default: voice.language or en-US)")

    s = gsub.add_parser("create", help="design a voice and store it in FILE")
    s.add_argument("file")
    voice_opts(s)
    s.add_argument("--force", action="store_true", help="overwrite an existing extensions.google")
    s = gsub.add_parser("get", help="show a server voice")
    s.add_argument("voice_id")
    s = gsub.add_parser("list", help="list server voices")
    s.add_argument("--type", default="prompted", help="voice type filter, or 'all'")
    s = gsub.add_parser("refresh", help="re-fetch sample + expiry into FILE(s)")
    s.add_argument("files", nargs="+")
    s = gsub.add_parser("update", help="recreate the voice (no PATCH), write FILE, delete the old id")
    s.add_argument("file")
    voice_opts(s)
    s = gsub.add_parser("synthesize", help="render TEXT with the voice stored in FILE")
    s.add_argument("file")
    s.add_argument("text", nargs="?")
    s.add_argument("--text-file")
    s.add_argument("--style", help="delivery style annotation, e.g. 'dry and tired'")
    s.add_argument("-o", "--output", required=True, help="output .wav")
    s.add_argument("--play", action="store_true", help="play the result after writing it")
    s = gsub.add_parser("delete", help="delete a server voice")
    s.add_argument("voice_id")
    s.add_argument("--vox", help="also remove extensions.google + sample from this .vox if it matches")
    s.add_argument("--yes", action="store_true", help="confirm the irreversible delete")
    gp.set_defaults(func=cmd_gemini)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (VoxError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
