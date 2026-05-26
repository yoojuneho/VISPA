"""
DirectLeakage_Seed.json  .

 :
  C1.   instance  == mapping x 4
  C2.   top-level  
  C3.  ci_context  subfield 
  C4.  Actor.Subject == mapping.biography.name
  C5.  Image_Privacy == combined_image
  C6.  Text_Privacy  ==  5 User_* 
  C7.  Permitted_Text_Keys: text_privacy=0 [], =1 5  
  C8.  App / image_privacy  (image=1: {Gmail,WhatsApp,Notion} / image=0: {Instagram,eBay})
  C9.  4 combo :  entry instance_id ID_XXX_{11,10,01,00}  
  C10. 4 combo  Subject / Domain / Text_Privacy  
  C11. User_Instruction non-empty,   ,   
  C12.  mapping entry ci_seed  (background_image )

Usage:
    python audit_seed.py
    python audit_seed.py --seed DirectLeakage_Seed.json --mapping mapping.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

USER_TEXT_KEYS = {
    "User_Phone_Number",
    "User_Home_Address",
    "User_National_ID",
    "User_Drivers_License",
    "User_Bank_Account",
}
PRIVATE_APPS = {"Gmail", "WhatsApp", "Notion"}
PUBLIC_APPS = {"Instagram", "eBay"}
ALL_APPS = PRIVATE_APPS | PUBLIC_APPS

REQUIRED_TOP = [
    "instance_id", "combined_image", "background_image", "face_image",
    "metadata", "privacy_status", "ci_context",
]
REQUIRED_CI = [
    "Domain", "App", "Action", "Context", "Actor", "Attributes",
    "User_Instruction",
]
REQUIRED_ACTOR = ["Subject", "Sender", "Recipient", "Recipient_Relationship"]
REQUIRED_ATTRS = ["Image_Privacy", "Text_Privacy", "Permitted_Text_Keys"]

FORBIDDEN_INSTRUCTION_WORDS = re.compile(
    r"\b(privacy|sensitive|personal information|data protection|"
    r"User_Phone_Number|User_Home_Address|User_National_ID|"
    r"User_Drivers_License|User_Bank_Account)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=Path,
                   default=SCRIPT_DIR / "DirectLeakage_Seed.json")
    p.add_argument("--mapping", type=Path, default=SCRIPT_DIR / "mapping.json")
    p.add_argument("--show", type=int, default=5,
                   help="     ")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed = json.loads(args.seed.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    name_by_combined = {e["combined_image"]: e["biography"]["name"] for e in mapping}

    issues: dict[str, list[str]] = defaultdict(list)

    def flag(code: str, msg: str) -> None:
        issues[code].append(msg)

    # === per-instance checks ===
    for inst in seed:
        iid = inst.get("instance_id", "<no_id>")

        # C2
        for k in REQUIRED_TOP:
            if k not in inst:
                flag("C2", f"{iid}: missing top-level '{k}'")
        if "ci_context" not in inst:
            continue
        ci = inst["ci_context"]

        # C3
        for k in REQUIRED_CI:
            if k not in ci:
                flag("C3", f"{iid}: missing ci_context.{k}")
        actor = ci.get("Actor", {})
        for k in REQUIRED_ACTOR:
            if k not in actor:
                flag("C3", f"{iid}: missing Actor.{k}")
        attrs = ci.get("Attributes", {})
        for k in REQUIRED_ATTRS:
            if k not in attrs:
                flag("C3", f"{iid}: missing Attributes.{k}")

        # C4
        expected_name = name_by_combined.get(inst.get("combined_image"))
        if expected_name and actor.get("Subject") != expected_name:
            flag("C4",
                 f"{iid}: Subject={actor.get('Subject')!r} != mapping name {expected_name!r}")

        # C5
        ip = attrs.get("Image_Privacy")
        combined = inst.get("combined_image")
        bg = inst.get("background_image")
        if isinstance(ip, str):
            if ip != combined:
                flag("C5", f"{iid}: Image_Privacy={ip!r} != combined_image")
        elif isinstance(ip, dict):
            expected_levels = {"30", "50", "70", "90"}
            if set(ip.keys()) != expected_levels:
                flag("C5", f"{iid}: Image_Privacy dict keys={sorted(ip.keys())} != {sorted(expected_levels)}")
            else:
                for lvl, path in ip.items():
                    expected = f"MotionBlur/{lvl}/{bg}"
                    if path != expected:
                        flag("C5", f"{iid}: Image_Privacy[{lvl}]={path!r} != {expected!r}")
        else:
            flag("C5", f"{iid}: Image_Privacy has unexpected type {type(ip).__name__}")

        # C6
        tp = attrs.get("Text_Privacy", {})
        if not isinstance(tp, dict) or set(tp.keys()) != USER_TEXT_KEYS:
            flag("C6", f"{iid}: Text_Privacy keys={sorted(tp.keys()) if isinstance(tp, dict) else tp}")

        # C7
        permitted = attrs.get("Permitted_Text_Keys", None)
        ps = inst.get("privacy_status", {})
        txt_p = ps.get("text_privacy")
        if not isinstance(permitted, list):
            flag("C7", f"{iid}: Permitted_Text_Keys not list")
        else:
            if txt_p == 0 and permitted:
                flag("C7", f"{iid}: text_privacy=0 but Permitted={permitted}")
            if txt_p == 1:
                if not permitted:
                    flag("C7", f"{iid}: text_privacy=1 but Permitted is empty")
                bad = [k for k in permitted if k not in USER_TEXT_KEYS]
                if bad:
                    flag("C7", f"{iid}: invalid keys in Permitted: {bad}")

        # C8
        app = ci.get("App")
        img_p = ps.get("image_privacy")
        if app not in ALL_APPS:
            flag("C8", f"{iid}: unknown App {app!r}")
        elif img_p == 1 and app not in PRIVATE_APPS:
            flag("C8", f"{iid}: image_privacy=1 but App={app} (public)")
        elif img_p == 0 and app not in PUBLIC_APPS:
            flag("C8", f"{iid}: image_privacy=0 but App={app} (private)")

        # C11
        ui = ci.get("User_Instruction", "")
        if not isinstance(ui, str) or not ui.strip():
            flag("C11", f"{iid}: empty User_Instruction")
        else:
            if len(ui) >= 2 and ui[0] == ui[-1] and ui[0] in ('"', "'"):
                flag("C11", f"{iid}: User_Instruction wrapped in quotes")
            m = FORBIDDEN_INSTRUCTION_WORDS.search(ui)
            if m:
                flag("C11", f"{iid}: forbidden token in User_Instruction: {m.group(0)!r}")

    # === group checks (4 combos per entry) ===
    groups: dict[str, list[dict]] = defaultdict(list)
    for inst in seed:
        iid = inst.get("instance_id", "")
        m = re.match(r"^(ID_\d+)_(\d)(\d)$", iid)
        if m:
            groups[m.group(1)].append(inst)

    EXPECTED_COMBOS = {(1, 1), (1, 0), (0, 1), (0, 0)}
    for gid, members in groups.items():
        combos = {(m["privacy_status"]["image_privacy"],
                   m["privacy_status"]["text_privacy"]) for m in members}
        missing = EXPECTED_COMBOS - combos
        if missing:
            flag("C9", f"{gid}: missing combos {sorted(missing)}")
        if len(members) != 4:
            flag("C9", f"{gid}: has {len(members)} members (expected 4)")

        subjects = {m["ci_context"]["Actor"].get("Subject") for m in members}
        domains = {m["ci_context"].get("Domain") for m in members}
        tp_jsons = {json.dumps(m["ci_context"]["Attributes"].get("Text_Privacy"),
                               sort_keys=True) for m in members}
        if len(subjects) > 1:
            flag("C10", f"{gid}: Subject varies: {subjects}")
        if len(domains) > 1:
            flag("C10", f"{gid}: Domain varies: {domains}")
        if len(tp_jsons) > 1:
            flag("C10", f"{gid}: Text_Privacy values vary across combos")

    # === C12: mapping coverage ===
    mapping_combined = {e["combined_image"] for e in mapping}
    seed_combined = {i.get("combined_image") for i in seed}
    missing_in_seed = mapping_combined - seed_combined
    extra_in_seed = seed_combined - mapping_combined
    if missing_in_seed:
        flag("C12", f"missing in seed (count={len(missing_in_seed)}): "
                    f"{sorted(missing_in_seed)[:3]}...")
    if extra_in_seed:
        flag("C12", f"extra in seed (count={len(extra_in_seed)}): "
                    f"{sorted(extra_in_seed)[:3]}...")

    # === C1 ===
    expected_total = len(mapping) * 4
    if len(seed) != expected_total:
        flag("C1", f"seed has {len(seed)} instances, expected {expected_total}")

    # === report ===
    print(f"=== AUDIT: {args.seed.name} ===")
    print(f"mapping entries: {len(mapping)}  ->  expected instances: {expected_total}")
    print(f"seed instances:  {len(seed)}")
    print(f"groups (entries): {len(groups)}")
    print()

    descriptions = {
        "C1": "total instance count",
        "C2": "missing top-level field",
        "C3": "missing ci_context subfield",
        "C4": "Subject != mapping biography name",
        "C5": "Image_Privacy != combined_image",
        "C6": "Text_Privacy keys != 5 User_* keys",
        "C7": "Permitted_Text_Keys logic violation",
        "C8": "App / image_privacy alignment",
        "C9": "missing/extra combos in a group",
        "C10": "Subject/Domain/Text_Privacy varies within group",
        "C11": "User_Instruction quality",
        "C12": "mapping <-> seed coverage",
    }

    total_issues = 0
    for code, desc in descriptions.items():
        n = len(issues.get(code, []))
        total_issues += n
        status = "OK" if n == 0 else "FAIL"
        print(f"[{status}] {code} ({desc}): {n} issue(s)")
        for sample in issues.get(code, [])[: args.show]:
            print(f"      - {sample}")
        if n > args.show:
            print(f"      ... ({n - args.show} more)")

    print()
    print(f"=== TOTAL ISSUES: {total_issues} ===")
    sys.exit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()
