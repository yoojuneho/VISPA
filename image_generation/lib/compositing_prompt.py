import json
import re

# ---------------------------------------------------------------------------
# Stage 1: Background analysis -> edit prompt generation
# ---------------------------------------------------------------------------

STAGE1_SYSTEM_PROMPT = """\
You are a computer vision expert specializing in realistic photo compositing for privacy research.
Analyze a background scene image and generate TWO DISTINCT privacy-leakage compositing scenarios,
each placing a person in a different position so they appear to have been accidentally captured -
not posing, not aware of the camera.

The two scenarios MUST differ in: (1) placement position within the frame and (2) the person's activity or body orientation.

Requirements for EACH scenario:
- FULL BODY with scene-appropriate clothing (rendered from a face reference image)
- Mid-ground distance (3-8 meters apparent distance)
- Natural, scene-appropriate activity
- Placed clearly OFF-CENTER: left third, right third, foreground corner, or background edge - NEVER the center of the frame
- Face must be at least partially visible: 3/4 angle, slight profile, or subtly turned head - facial features must be recognizable even though the person is unaware of the camera; fully back-facing poses are NOT acceptable
- NO posed gestures (V-signs, waving, pointing at camera, smiling at lens)
- Partial occlusion by furniture or objects encouraged for realism

Scene-specific activity reference:
- airport_terminal / train_station: pulling luggage, checking phone, waiting in queue at an angle
- art_gallery / museum: examining artwork sideways, reading exhibit label with face in partial profile
- conference_room / office / meeting_room: seated at side of table looking at screen, writing notes with face turned
- library: browsing shelves at an angle, reading at a desk with face partially turned
- cafeteria / coffee_shop / food_court: seated eating, looking sideways or down at a menu
- bedroom: seated at a desk looking at laptop in profile, standing near a window with face visible
- corridor / hallway: walking past at an angle, glancing sideways mid-stride
- gym / fitness_center: using side equipment with face in profile, stretching while looking sideways
- classroom / lecture_room: seated writing, glancing toward the board showing a 3/4 face angle
- waiting_room / lobby: seated glancing sideways, reading with face slightly turned outward
- bathroom: at sink in 3/4 profile, drying hands while slightly turning

Output ONLY a valid JSON array of exactly 2 scenario objects (no markdown, no extra text):
[
  {
    "scenario_index": 0,
    "scene_description": "<2-3 sentence factual description of the scene>",
    "scene_category": "<single lowercase category, e.g. art_gallery>",
    "placement_hint": "<explicit off-center position, e.g. left foreground partially behind a column>",
    "edit_prompt": "<complete Qwen-Image-Edit prompt - must start exactly with: Preserve the exact background from image1 without any alteration.>",
    "negative_prompt": "<comma-separated visual artifacts and behaviors to avoid>"
  },
  {
    "scenario_index": 1,
    "scene_description": "...",
    "scene_category": "...",
    "placement_hint": "<a different off-center position from scenario 0>",
    "edit_prompt": "...",
    "negative_prompt": "..."
  }
]

Each edit_prompt must:
1. Begin exactly with: "Preserve the exact background from image1 without any alteration."
2. Use the facial identity from image2 to render a complete full-body person
3. Specify exact clothing, body pose, activity, and an explicit OFF-CENTER position (e.g., "in the left third of the frame", "near the right edge")
4. Describe face orientation as naturally angled: 3/4 profile or slight turn - features visible but person not looking at camera
5. Be detailed and specific (3-5 sentences total)

Each negative_prompt must include: centered composition, person at center of frame, facing camera directly, direct eye contact with camera, posing for camera, V-sign, waving at camera, back of head only, blurry face, face fully hidden, CGI artifacts, floating figure, lighting mismatch\
"""

STAGE1_USER_PROMPT = (
    "Analyze this background scene and generate two distinct compositing scenarios for a person "
    "accidentally captured in the frame. Output only the JSON array."
)

# ---------------------------------------------------------------------------
# Stage 3: Composited image quality validation
# ---------------------------------------------------------------------------

STAGE3_SYSTEM_PROMPT = """\
You are a quality-control expert evaluating photo compositing realism for privacy research.

Assess whether the composited image looks natural and credible. A good composite has:
1. Background intact - no distortion, blurring, or unintended alteration of background elements
2. A clearly visible person who is naturally integrated (lighting, shadow, perspective consistent)
3. The person appears accidentally captured - NOT posing, NOT making direct eye contact with the camera; a natural 3/4 face angle or slight profile is acceptable and expected
4. Scene-appropriate behavior and attire
5. The person's face is at least partially visible and recognizable (fully back-facing with no face visible is a defect)
6. No uncanny or CGI-like artifacts on the person

Output ONLY a valid JSON object (no markdown, no extra text):
{
  "decision": "keep",
  "reason": "<brief explanation of your decision>",
  "background_preserved": true,
  "person_detected": true,
  "natural_placement": true,
  "scene_appropriate": true,
  "not_posing": true
}

Use "reject" for decision if ANY of the following is true:
- Background is visibly altered or distorted
- No person is present or the person is not recognizable
- Person looks pasted-in (lighting/perspective mismatch, floating, hard edges)
- Person is clearly posing for or making direct eye contact with the camera
- Face is completely hidden (back of head only) with no facial features visible
- Obvious CGI artifacts or blending errors\
"""

STAGE3_USER_PROMPT = (
    "Evaluate this composited image. Is the person naturally integrated into the background scene "
    "as if accidentally captured, with their face at least partially visible? Output only the JSON."
)

# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _extract_json_from_text(text: str) -> str:
    text = text.strip()
    block = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if block:
        return block.group(1).strip()
    arr_start = text.find("[")
    obj_start = text.find("{")
    if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
        arr_end = text.rfind("]")
        if arr_end > arr_start:
            return text[arr_start : arr_end + 1]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_stage1_response(raw_text: str) -> dict:
    """[Deprecated] Parse Stage 1 single-scenario response. Use parse_stage1_scenarios instead."""
    extracted = _extract_json_from_text(raw_text)
    try:
        data = json.loads(extracted)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        parse_error = False
    except json.JSONDecodeError:
        data = {
            "scene_description": "",
            "scene_category": "other",
            "placement_hint": "mid-ground",
            "edit_prompt": "",
            "negative_prompt": "",
        }
        parse_error = True
    data["parse_error"] = parse_error
    data["raw_response"] = raw_text
    return data


_SCENARIO_FIELDS = (
    "scene_description", "scene_category", "placement_hint",
    "edit_prompt", "negative_prompt",
)


def parse_stage1_scenarios(raw_text: str) -> list[dict]:
    """Parse Stage 1 VLM response that contains 2 scenario objects.

    Returns a list of exactly 2 dicts, each with scenario_index, scene_description,
    scene_category, placement_hint, edit_prompt, negative_prompt, parse_error, raw_response.
    On parse failure fills both slots with empty error dicts.
    """
    extracted = _extract_json_from_text(raw_text)
    try:
        data = json.loads(extracted)
        if isinstance(data, list) and len(data) >= 2:
            scenarios = []
            for i, item in enumerate(data[:2]):
                scenarios.append({
                    "scenario_index": i,
                    **{f: item.get(f, "") for f in _SCENARIO_FIELDS},
                    "parse_error": False,
                    "raw_response": raw_text,
                })
            return scenarios
    except json.JSONDecodeError:
        pass
    return [
        {
            "scenario_index": i,
            "scene_description": "", "scene_category": "other",
            "placement_hint": "", "edit_prompt": "", "negative_prompt": "",
            "parse_error": True, "raw_response": raw_text,
        }
        for i in range(2)
    ]


def parse_stage3_response(raw_text: str) -> dict:
    """Parse Stage 3 VLM response. Returns dict with decision and detail flags."""
    extracted = _extract_json_from_text(raw_text)
    try:
        data = json.loads(extracted)
        decision = str(data.get("decision", "")).strip().lower()
        if decision not in ("keep", "reject"):
            decision = "reject"
        data["decision"] = decision
        parse_error = False
    except json.JSONDecodeError:
        data = {
            "decision": "reject",
            "reason": "Failed to parse VLM response",
            "background_preserved": False,
            "person_detected": False,
            "natural_placement": False,
            "scene_appropriate": False,
            "not_posing": False,
        }
        parse_error = True
    data["parse_error"] = parse_error
    data["raw_response"] = raw_text
    return data


# ---------------------------------------------------------------------------
# Pairwise visible-person choice evaluation
# ---------------------------------------------------------------------------

PAIR_PERSON_CHOICE_SYSTEM_PROMPT = """\
You are a strict visual evaluator for image-pair benchmarking.
You will receive exactly two images, labeled Image 1 and Image 2.
Your task is to decide which image contains a visible real person.

Important rules:
- Use only visible image content.
- Ignore filenames, paths, ordering priors, dataset conventions, and any assumption about which image is original or edited.
- A visible person may be indicated by a human face, head, upper body, full body, or a clearly visible real human figure.
- Choose "None" only if neither image contains a visible real person.
- Return JSON only with no markdown and no extra text.
"""

PAIR_PERSON_CHOICE_USER_PROMPT = """\
Evaluate the two images in order.

Return JSON only in this schema:
{"decision": "1" | "2" | "None", "reason": "<short visual justification>"}

Decision rules:
- "1": Image 1 contains a visible real person and Image 2 does not.
- "2": Image 2 contains a visible real person and Image 1 does not.
- "None": neither image contains a visible real person.

Use a short visual reason only. Do not mention filenames, paths, metadata, or assumptions about generation history.
"""


def parse_pair_person_choice_response(raw_text: str) -> dict:
    """Parse pairwise person-choice VLM response.

    Returns dict with decision in {"1", "2", "None"}, best-effort reason,
    parse_error, and raw_response.
    """
    extracted = _extract_json_from_text(raw_text)
    try:
        data = json.loads(extracted)
        decision = str(data.get("decision", "None")).strip()
        if decision not in ("1", "2", "None"):
            decision = "None"
        reason = str(data.get("reason", "no_reason")).strip() or "no_reason"
        return {
            "decision": decision,
            "reason": reason,
            "parse_error": False,
            "raw_response": raw_text,
        }
    except json.JSONDecodeError:
        decision_match = re.search(r'"decision"\s*:\s*"(1|2|None)"', extracted, re.IGNORECASE)
        reason_match = re.search(r'"reason"\s*:\s*"([^\"]*)"', extracted, re.IGNORECASE)
        decision = decision_match.group(1) if decision_match else "None"
        if decision.lower() == "none":
            decision = "None"
        reason = reason_match.group(1).strip() if reason_match else "unparseable_response"
        return {
            "decision": decision,
            "reason": reason or "unparseable_response",
            "parse_error": True,
            "raw_response": raw_text,
        }


# ---------------------------------------------------------------------------
# Monitor reflection compositing quality filter (1-shot)
