"""Fresh, explicitly synthetic controls for the registered final supplement."""

import copy


# Every theme stays in its previously assigned split; wording is newly authored.
THEMES = {
    "authored-v3-en-accessible-museum": ("Place the", "tactile card beside the display.", "The raised dots identify the route.", "The guide is ready.", "The guide is raedy."),
    "authored-v3-en-community-mapping": ("Draw the", "footpath around the flooded field.", "The dotted line marks a public path.", "The map is complete.", "The map is compelte."),
    "authored-v3-en-oral-archive": ("Store the", "interview copy in the archive box.", "The speaker approved this recording.", "The copy is complete.", "The copy is compelte."),
    "authored-v3-en-radio-observatory": ("Point the", "receiver toward the northern ridge.", "The reference signal remains stable.", "The receiver is ready.", "The receiver is raedy."),
    "authored-v3-en-repair-cafe": ("Slide the", "protective cover over the switch.", "The plug must stay disconnected.", "The cover is secure.", "The cover is seucre."),
    "authored-v3-en-river-laboratory": ("Lower the", "sample holder into the clear water.", "The upstream reading is our control.", "The holder is empty.", "The holder is emtpy."),
    "authored-v3-en-ferry-training": ("Fasten the", "boarding gate before departure.", "The deck officer checks the latch.", "The gate is secure.", "The gate is seucre."),
    "authored-v3-en-natural-history-tour": ("Compare the", "shell outline with the drawing.", "The spiral grows wider near the rim.", "The drawing is complete.", "The drawing is compelte."),
    "authored-v3-ko-archive-catalog": ("이 자료의", "기증 기록을 목록에 남깁니다.", "기증 날짜는 원본 기록을 따릅니다.", "자료를 보관합니다.", "자료를 보관합니디."),
    "authored-v3-ko-community-radio": ("다음 순서의", "인터뷰 음성을 먼저 확인합니다.", "출연자는 공개 범위를 정했습니다.", "음성을 확인합니다.", "음성을 확인합니디."),
    "authored-v3-ko-pottery-studio": ("작은 그릇의", "굽 부분을 천천히 다듬습니다.", "유약이 닿는 면은 그대로 둡니다.", "그릇을 말립니다.", "그릇을 말립니디."),
    "authored-v3-ko-rooftop-garden": ("이 화분의", "흙 표면을 손가락으로 확인합니다.", "그늘에 둔 화분은 아직 촉촉합니다.", "화분을 옮깁니다.", "화분을 옮깁니디."),
    "authored-v3-ko-subtitle-class": ("발표 장면의", "첫 자막을 화면 아래에 놓습니다.", "고유명사의 표기는 원문을 따릅니다.", "자막을 저장합니다.", "자막을 저장합니디."),
    "authored-v3-ko-weather-balloon": ("관측 장비의", "연결선을 덮개 안으로 넣습니다.", "기압계는 지상 값을 기록했습니다.", "장비를 준비합니다.", "장비를 준비합니디."),
    "authored-v3-ko-seed-library": ("반납된 종자의", "채종 연도를 봉투에 적습니다.", "발아율은 품종마다 다를 수 있습니다.", "종자를 분류합니다.", "종자를 분류합니디."),
    "authored-v3-ko-tool-lending": ("대여할 공구의", "손잡이를 빌리기 전에 살펴봅니다.", "반납일은 대여 기록에 적혀 있습니다.", "공구를 점검합니다.", "공구를 점검합니디."),
}


def text(key, value, ms=1600):
    return {"key": key, "text": value, "ms": ms}


def gap(key="gap", ms=1600):
    return {"key": key, "ms": ms}


def plain(slot, paragraphs, focus, primary, reason, **kwargs):
    return {**slot, "paragraphs": paragraphs, "focus": focus, "primary": primary,
            "rationale": reason, **kwargs}


def long_paragraph(slot, positive):
    en = slot["language"] == "en"
    head = (
        "The record follows the route from the old entrance through the covered corridor past the wooden cabinets and around the central courtyard toward the wide doorway beside the reading room with its low shelves and large windows overlooking the garden and the narrow passage behind the building that connects the workroom to the quiet storage area near the rear steps"
        if en else
        "기록은 오래된 입구에서 시작해 지붕이 있는 복도를 지나 나무 보관함 옆을 따라 중앙 마당을 돌아 낮은 책장과 정원이 보이는 큰 창문이 있는 열람실 옆의 넓은 출입구를 향하고 건물 뒤편의 좁은 통로를 따라 작업실과 뒤쪽 계단 근처의 조용한 보관 공간을 연결하는 길의 바닥 재료와 벽면의 색상과 창문의 위치와 문의 방향과 각 구역의 조명 상태와 이동 순서를 차례대로 설명하는 안내문의"
    )
    if slot["split"] == "dev":
        head = (
            "The tour follows the contour of the shoreline from the shallow inlet beyond the scattered pebbles along the sandy strip below the low cliffs through the sheltered bend beside the nesting ground and toward the sheltered pool behind the headland where the raised boardwalk returns to the landing place with a narrow path crossing the flat rocks near the boat shed"
            if en else
            "설명은 종자를 받는 창구에서 시작해 분류대 옆의 통로를 따라 봉투를 놓는 선반을 지나 온도와 습도를 확인하는 보관실 앞까지 이어지고 대여 공구를 반납하는 작업대 뒤의 작은 세척 공간과 점검을 마친 물건을 두는 건조대와 기록을 작성하는 책상을 연결하는 동선의 바닥 표시와 안내판의 위치와 각 구역의 이용 순서와 확인해야 하는 항목을 차례대로 소개하는 안내문의"
        )
    if slot["split"] == "train":
        substitutions = {
            "authored-v3-en-accessible-museum": (("corridor", "gallery"), ("cabinets", "display cases"), ("courtyard", "exhibition hall"), ("reading room", "touch display"), ("shelves", "panels"), ("garden", "sculpture court"), ("workroom", "visitor workshop")),
            "authored-v3-en-community-mapping": (("corridor", "footpath"), ("cabinets", "notice boards"), ("courtyard", "public square"), ("reading room", "map room"), ("shelves", "railings"), ("garden", "canal"), ("workroom", "survey office")),
            "authored-v3-en-oral-archive": (("corridor", "archive aisle"), ("cabinets", "tape cabinets"), ("courtyard", "catalogue desk"), ("reading room", "listening room"), ("shelves", "reel shelves"), ("garden", "delivery yard"), ("workroom", "digitizing room")),
            "authored-v3-en-radio-observatory": (("corridor", "instrument corridor"), ("cabinets", "receiver racks"), ("courtyard", "antenna base"), ("reading room", "control room"), ("shelves", "equipment bays"), ("garden", "northern ridge"), ("workroom", "calibration room")),
            "authored-v3-en-repair-cafe": (("corridor", "workshop aisle"), ("cabinets", "tool chests"), ("courtyard", "repair bench"), ("reading room", "testing area"), ("shelves", "parts trays"), ("garden", "service yard"), ("workroom", "soldering station")),
            "authored-v3-en-river-laboratory": (("corridor", "sampling corridor"), ("cabinets", "sample coolers"), ("courtyard", "washing station"), ("reading room", "analysis room"), ("shelves", "bottle racks"), ("garden", "river bank"), ("workroom", "sensor laboratory")),
            "authored-v3-ko-archive-catalog": (("복도", "자료실 통로"), ("보관함", "기증 기록함"), ("마당", "분류 작업대"), ("열람실", "목록실"), ("정원", "반입 공간"), ("작업실", "촬영실")),
            "authored-v3-ko-community-radio": (("복도", "방송국 복도"), ("보관함", "음원 보관함"), ("마당", "공개 스튜디오"), ("열람실", "조정실"), ("정원", "녹음 부스"), ("작업실", "편집실")),
            "authored-v3-ko-pottery-studio": (("복도", "공방 통로"), ("보관함", "도구 보관함"), ("마당", "유약 작업대"), ("열람실", "건조실"), ("정원", "가마 앞 공간"), ("작업실", "물레 작업장")),
            "authored-v3-ko-rooftop-garden": (("복도", "옥상 통로"), ("보관함", "모종 보관함"), ("마당", "물주기 공간"), ("열람실", "온실"), ("정원", "텃밭"), ("작업실", "분갈이 작업장")),
            "authored-v3-ko-subtitle-class": (("복도", "강의실 복도"), ("보관함", "영상 자료함"), ("마당", "실습 공간"), ("열람실", "상영실"), ("정원", "촬영 공간"), ("작업실", "자막 편집실")),
            "authored-v3-ko-weather-balloon": (("복도", "관측소 통로"), ("보관함", "센서 보관함"), ("마당", "발사 준비대"), ("열람실", "관제실"), ("정원", "관측장"), ("작업실", "장비 점검실")),
        }
        for before, after in substitutions[slot["source_group"]]:
            head = head.replace(before, after)
    elif slot["source_group"].endswith("ferry-training"):
        for before, after in (("tour", "boarding inspection"), ("shoreline", "harbour"), ("nesting ground", "loading ramp"), ("sheltered pool", "mooring point"), ("raised boardwalk", "passenger gangway")):
            head = head.replace(before, after)
    elif slot["source_group"].endswith("tool-lending"):
        for before, after in (("종자", "수공구"), ("봉투", "운반 상자"), ("온도와 습도", "손잡이와 날 상태"), ("보관실", "공구실")):
            head = head.replace(before, after)
    words = head.split()
    items = [text(f"w{i}", word, 180) for i, word in enumerate(words)]
    tail = ("written description." if en else "문장입니다.")
    items.append(text("tail", tail, 1100))
    if positive:
        items.extend([text("new", "A separate inspection starts here." if en else "다음 점검은 여기서 시작합니다.", 2200),
                      text("next", "We now check the warning sign." if en else "이제 안내 표지를 확인합니다.", 2200)])
        focus = "tail"
        primary = {"action": "split_paragraph", "targets": ["new"]}
        reason = "The native window crops the long preceding paragraph, so a merge cannot be justified. A visible completed description then gives way to an explicit new inspection; the local split preserves all text and timing and produces cues over one second."
    else:
        focus = f"w{len(words) // 2}"
        primary = {"action": "no_action"}
        reason = "The visible window is an interior fragment of one continuous route description. No paragraph boundary exists at the cursor, hidden wording cannot support a merge, and the visible fragment contains no typo, abandoned instruction, demonstrated pacing preference, or clearly necessary sentence split. Independent review may defer this authored preservation claim."
    return plain(slot, [items], focus, primary, reason, full_context_control=True)


def contract(slot):
    a, b, neutral, correct, typo = THEMES[slot["source_group"]]
    family, en = slot["family"], slot["language"] == "en"
    no = {"action": "no_action"}
    common = "All words are intentional and current; no deletion, translation, spelling change, or pacing edit is justified elsewhere in the visible window. "
    if family in ("paragraph_cropped", "cropped_paragraph_local_split"):
        return long_paragraph(slot, family.endswith("local_split"))
    if family == "useful_merge":
        return plain(slot, [[text("a", a, 1200)], [text("b", b, 2300)]], "a",
                     {"action": "merge_paragraph", "targets": ["b"]},
                     "The first complete visible caption stops inside a dependent phrase. Joining the two known-speaker captions repairs that fracture within the caption limits.")
    if family == "boundary_absent":
        return plain(slot, [[text("a", a), text("b", b, 2300)]], "b", no,
                     common + "The dependent phrase is already together in one paragraph; no boundary is available or useful to merge.")
    if family == "speaker_unknown":
        return plain(slot, [{"speaker": None, "items": [text("a", a, 1200)]},
                            {"speaker": None, "items": [text("b", b, 2300)]}], "b", no,
                     common + "Both speakers are genuinely unavailable in this synthetic input. The apparent phrase continuation does not establish speaker compatibility and no other repair is supported.")
    if family == "speaker_changes":
        return plain(slot, [{"speaker": "host", "items": [text("a", neutral, 2700)]},
                            {"speaker": "guest", "items": [text("b", correct, 2000)]}], "b", no,
                     common + "Two complete utterances from different known speakers retain their useful turn boundary.")
    if family == "language_changes":
        other = "Please check the label before continuing." if not en else "다음 단계 전에 표기를 확인해 주세요."
        return plain(slot, [{"language": slot["language"], "items": [text("a", neutral, 2700)]},
                            {"language": "ko" if en else "en", "items": [text("b", other, 2300)]}], "a", no,
                     common + "The two complete captions switch languages deliberately and keep that boundary; caption cleanup is not translation.")
    if family == "merge_duration_overflow":
        return plain(slot, [[text("a", neutral, 4100)], [text("b", correct, 3800)]], "b", no,
                     common + "Each caption is a coherent thought. Merging would exceed the seven-second merged cue limit and has no independent benefit.")
    if family == "merge_text_overflow":
        first = neutral + (" The position is shown in the printed instructions." if en else " 해당 위치는 인쇄된 안내문에도 같은 순서로 설명되어 있습니다.")
        second = correct + (" We keep the original markings visible throughout the check." if en else " 점검이 끝날 때까지 기존 표시는 지우지 않고 확인할 수 있도록 남겨 둡니다.")
        return plain(slot, [[text("a", first, 3200)], [text("b", second, 3200)]], "b", no,
                     common + "The two whole meaningful captions exceed the merged 84 UTF-16 character limit together and already express separate complete observations.")
    if family in ("valid_merge_unnecessary", "split_boundary_exists"):
        return plain(slot, [[text("a", neutral, 2600)], [text("b", correct, 2200)]],
                     "b" if family == "split_boundary_exists" else "a", no,
                     common + "The two short complete thoughts already have a sensible paragraph boundary; another split repeats it and a legal merge would be gratuitous.")
    if family == "already_corrected_text":
        return plain(slot, [[text("a", typo, 2200)]], "a", no,
                     common + "The actual latest transaction already replaced the typo. The current display is correct; repeating the correction is unchanged.",
                     history=[{"action": "correct_text", "targets": ["a"], "text": correct}])
    if family == "gap_without_deletable_words":
        wait = "Pause to check this observation: " + neutral if en else "다음 내용을 확인하는 동안 기다립니다. " + neutral
        done = "The check is complete. " + correct if en else "확인을 마쳤습니다. " + correct
        return plain(slot, [[text("a", wait, 3000), gap(ms=1200)], [text("b", done, 1800)]], "gap", no,
                     common + "The non-text interval is an explicitly meaningful viewing wait, contains no deletable words, and is not an unwanted interruption.")
    if family == "already_shortened_gap":
        return plain(slot, [[text("a", a), gap(ms=1900), text("b", b, 2300)]], "gap", no,
                     common + "A real preceding trim has already reduced this exact phrase gap to 400ms. No new evidence supports another pacing intervention.",
                     history=[{"action": "trim_silence", "targets": ["gap"], "keep_ms": 400}])
    if family == "unknown_speaker_elsewhere_correction":
        return plain(slot, [{"speaker": None, "items": [text("a", neutral, 2700)]},
                            {"speaker": None, "items": [text("b", typo, 2200)]}], "a",
                     {"action": "correct_text", "targets": ["b"], "text": correct},
                     "An ordinary-word transposition or malformed Korean ending has one clear display correction. Missing speakers prohibit speculative merging but do not hide the visible spelling error.")
    if family == "absent_boundary_elsewhere_split":
        return plain(slot, [[text("a", a), text("b", b, 2300), text("c", neutral, 2600)]], "b",
                     {"action": "split_paragraph", "targets": ["c"]},
                     "The cursor is inside a dependent phrase with no merge boundary. The later completed instruction changes to a separate statement; splitting there gives two coherent cues without changing speech.")
    raise ValueError("Unimplemented contract condition: " + family)


def rare(slot):
    a, b, neutral, correct, typo = THEMES[slot["source_group"]]
    action, family = slot["action"], slot["family"]
    en, useful = slot["language"] == "en", slot["necessity"] == "useful"
    no = {"action": "no_action"}
    if action == "delete_span":
        index = ("multi_item_false_start", "false_start_correction", "abandoned_clause").index(family)
        if useful:
            wrong = ("Use the first", "Start on the left", "I was going to say the top")[index] if en else ("첫 번째 것을", "왼쪽부터", "위쪽이라고 하려다가")[index]
            repair = ("use the second one.", "start on the right.", "use the lower one.")[index] if en else ("두 번째 것을 씁니다.", "오른쪽부터 시작합니다.", "아래쪽을 씁니다.")[index]
            lead = "For this step," if en else "이번 단계에서는"
            return plain(slot, [[text("lead", lead, 1200), text("bad", wrong, 850), gap("breath", 140),
                                text("marker", "no, sorry," if en else "아니,", 500), text("repair", repair, 2100)],
                                [text("end", neutral, 2500)]], ("end", "bad", "lead")[index],
                         {"action": "delete_span", "targets": ["bad", "breath", "marker"]},
                         "The immediately following self-correction explicitly abandons the first instruction. Delete its contiguous abandoned words, intervening breath and repair marker, retaining the final choice and all context. Symbolic source timing is not listening evidence.")
        words = ("Use the first one for practice and the second one for the final check.",
                 "Start on the left, then work toward the right.",
                 "I was going to use the top one, but the lower one fits this case.") if en else (
                     "첫 번째 것은 연습에 쓰고 두 번째 것은 마지막 확인에 씁니다.",
                     "왼쪽부터 시작한 다음 오른쪽으로 진행합니다.",
                     "위쪽을 쓰려고 했지만 이번 경우에는 아래쪽이 맞습니다.")
        return plain(slot, [[text("a", words[index], 4100)], [text("b", neutral, 2500)]], "a", no,
                     "The first/second or left/right alternatives express a sequence, comparison, or explicit reason for a changed decision. Removing the apparent false start would erase meaningful content. Both captions and wording are already coherent; no useful pacing edit is established.")
    if action == "merge_paragraph":
        index = ("merge_incomplete_phrase", "merge_sentence_fragment", "merge_modifier_fragment").index(family)
        if useful:
            parts = (a, b) if index == 0 else (("The next task is to", (a + " " + b)[0].lower() + (a + " " + b)[1:]) if en and index == 1
                                            else ("다음 순서는", a + " " + b) if not en and index == 1
                                            else ("For this demonstration," if en else "이번 시연에서는", a + " " + b))
            return plain(slot, [[text("a", parts[0], 1200)], [text("b", parts[1], 2700)]], "a" if index != 1 else "b",
                         {"action": "merge_paragraph", "targets": ["b"]},
                         "Both complete visible captions share speaker and language. The first ends with an unfinished construction whose completion is the second; merging restores one concise thought inside the profile limits.")
        retain_first, retain_second = neutral, correct
        if index == 1:
            retain_first, retain_second = ("Is everything ready?", "Yes. " + correct) if en else ("준비를 모두 마쳤나요?", "네. " + correct)
        if index == 2:
            retain_first, retain_second = ("First, " + a.lower() + " " + b, "Afterward, " + correct[0].lower() + correct[1:]) if en else ("먼저 " + a + " " + b, "그다음에 " + correct)
        return plain(slot, [[text("a", retain_first, 2600)], [text("b", retain_second, 2000)]], "a" if index == 0 else "b", no,
                     "These are two independent complete thoughts, with a useful existing caption boundary. No typo, abandoned speech, or motivated gap change exists; technical merge legality alone is not a reason to edit.")
    keep = 400 if slot["realization"] == 1 else 550
    prior = {"action": action, "targets": ["hg"]}
    primary = {"action": action, "targets": ["target"]}
    if action == "trim_silence":
        prior["keep_ms"] = primary["keep_ms"] = keep
    history = [prior]
    paragraphs = [[text("ha", "Keep the next section" if en else "다음 구간은", 1600),
                   gap("hg", 1800 if action == "trim_silence" else 600),
                   text("hb", "in the same order." if en else "같은 순서로 둡니다.", 1700)]]
    if family in ("mixed_history_current_wording", "mixed_history_early_target"):
        paragraphs.append([text("typo", typo, 2300)])
        history.append({"action": "correct_text", "targets": ["typo"], "text": correct})
    if useful:
        current = [text("a", a, 1700), gap("target", 2300 if action == "trim_silence" else 700), text("b", b, 2300)]
        reason = "The actual prior pacing transaction establishes this retention style for a within-phrase interruption. The current gap breaks the same kind of tightly linked phrase. A reversible cut preview preserves the whole instruction; acoustic state remains unknown."
    else:
        wait = "Wait while the group checks this: " + neutral if en else "참가자가 다음 내용을 확인하는 동안 기다립니다. " + neutral
        done = "The check is complete. " + correct if en else "확인을 마쳤습니다. " + correct
        current = [text("a", wait, 2800), gap("target", 2300 if action == "trim_silence" else 700)]
        reason = "The current gap is an explicit observation wait, unlike the earlier edited interruption inside a phrase. The completed instruction and report need all their meaning and existing boundaries. Prior pacing is not a blanket command to cut every interval."
        primary = no
    if family == "mixed_history_early_target":
        paragraphs.insert(0, current)
        if not useful:
            paragraphs.insert(1, [text("done", done, 2500)])
        focus = "typo"
    else:
        paragraphs.append(current)
        if not useful:
            paragraphs.append([text("done", done, 2500)])
        focus = "target" if family in ("executed_pacing_history", "matched_history_pacing_choice") else "ha"
    if family == "multiple_gaps_selective_trim":
        paragraphs.insert(1, [text("observe", "Listen for the signal." if en else "신호음을 들어 보세요.", 2000), gap("signal", 600)])
        annotations = [["signal", "audio_event"]]
        reason += " The other gap is a known signal and must stay."
    else:
        annotations = []
    return plain(slot, paragraphs, focus, primary, reason, history=history, annotations=annotations)


def history_fixture(episode_slots):
    """Actual eight-transition episodes, with fresh split-specific roles and names."""
    distractors = {
        "en": [["The label is visble.", "The label is visible."], ["The record is compelte.", "The record is complete."],
               ["The surface is celan.", "The surface is clean."], ["The hinge is seucre.", "The hinge is secure."],
               ["The packet is raedy.", "The packet is ready."], ["The shelf is emtpy.", "The shelf is empty."],
               ["The marker is staight.", "The marker is straight."], ["The handle is attatched.", "The handle is attached."]],
        "ko": [["표기를 확인합니디.", "표기를 확인합니다."], ["기록을 보관합니디.", "기록을 보관합니다."],
               ["표면을 닦습니디.", "표면을 닦습니다."], ["덮개를 닫습니디.", "덮개를 닫습니다."],
               ["봉투를 준비합니디.", "봉투를 준비합니다."], ["선반을 비웁니디.", "선반을 비웁니다."],
               ["표시를 맞춥니디.", "표시를 맞춥니다."], ["손잡이를 고정합니디.", "손잡이를 고정합니다."]],
    }
    neutral = {
        "en": ["The front table is clear.", "Each visitor has a marked place.",
               "The instruction sheet shows the order.", "The storage shelf is already labeled.",
               "The room has two separate exits.", "Keep the main walkway open.",
               "The spare supplies remain inside.", "Visitors may ask for assistance.",
               "The guide explains each step.", "Everyone can see the demonstration.",
               "Completed items go on the tray.", "The next group arrives after lunch.",
               "The signs remain in their places.", "The demonstration uses ordinary materials.",
               "The notes stay beside the display.", "Each activity has a clear purpose.",
               "Participants return materials after use.", "The session ends at the doorway."],
        "ko": ["앞쪽 작업대는 깨끗하게 비워 두었습니다.", "참가자마다 정해진 자리가 따로 있습니다.",
               "안내문에는 오늘의 활동 순서가 적혀 있습니다.", "보관 선반에는 각각의 이름표가 붙어 있습니다.",
               "이 공간에는 출구가 두 곳 있습니다.", "가운데 통로는 항상 비워 두세요.",
               "남은 준비물은 보관함 안에 있습니다.", "참가자는 언제든지 도움을 요청할 수 있습니다.",
               "안내자는 각 단계를 차례대로 설명합니다.", "모두가 시연 장면을 볼 수 있습니다.",
               "완성한 물건은 작업대 위에 놓습니다.", "다음 모임은 점심시간 이후에 시작합니다.",
               "안내 표지는 정해진 위치에 둡니다.", "이번 시연에는 일반적인 재료를 사용합니다.",
               "기록지는 전시물 옆에 그대로 둡니다.", "각 활동에는 분명한 목적이 있습니다.",
               "사용한 준비물은 모두 제자리에 돌려놓습니다.", "안내는 출입구 앞에서 마무리합니다."],
    }
    names = {
        ("train", "en", 1): ("Melan", "Melynn", "Melann", "restorer"),
        ("train", "en", 2): ("Coren", "Coryn", "Corenn", "technician"),
        ("dev", "en", 1): ("Nevan", "Nevyn", "Nevann", "navigator"),
        ("train", "ko", 1): ("서린", "서린아", "서리나", "복원 담당자"),
        ("train", "ko", 2): ("도윤", "도유나", "도윤아", "관측 담당자"),
        ("dev", "ko", 1): ("하린", "하리나", "하린아", "종자 담당자"),
    }
    fixture = {"distractors": distractors, "neutral_body": neutral, "episodes": [],
               "provenance": {"actor": "synthetic", "author_id": "final_dataset_agent", "model_predictions_consulted": False,
                              "observed_human_actions": False, "final_test_bodies_read": False, "timing": "authored_symbolic_not_audio"}}
    for slot in episode_slots:
        old, middle, latest, role = names[(slot["split"], slot["language"], slot["realization"])]
        en = slot["language"] == "en"
        mechanism = slot["mechanism"]
        def role_sentence(name):
            if mechanism == "unrelated_role_history":
                return f"The guest musician is {name}." if en else f"초대 연주자는 {name}입니다."
            return f"Today's only {role} is {name}." if en else f"오늘 유일한 {role}는 {name}입니다."

        def current(name):
            return f"Our {role} {name} checks the final entry." if en else f"{role} {name} 씨가 마지막 기록을 확인합니다."
        if (slot["language"], slot["split"], slot["realization"], mechanism) == ("ko", "train", 1, "latest_correction_supersedes_older"):
            old, middle, latest = "서린", "세린", "새린"
        key = "role"
        if mechanism == "unrelated_role_history":
            current_value = f"Our {role} {old} is not the guest musician." if en else f"{role} {old} 씨는 초대 연주자와 다른 사람입니다."
            critical = {"7": {"key": key, "before": role_sentence(old), "after": role_sentence(middle)}}
            replacement, hidden, expected = None, [middle], ("no_action",) * 3
        elif mechanism == "latest_correction_supersedes_older":
            critical = {"2": {"key": key, "before": role_sentence(old), "after": role_sentence(middle)},
                        "7": {"key": key, "before": role_sentence(middle), "after": role_sentence(latest)}}
            current_value, replacement, hidden = current(middle), current(latest), [old, latest]
            expected = ("no_action", "correct_text", "correct_text")
        else:
            position = "7" if mechanism == "evidence_in_last_three" else "2"
            critical = {position: {"key": key, "before": role_sentence(old), "after": role_sentence(middle)}}
            current_value, replacement, hidden = current(old), current(middle), [middle]
            expected = ("no_action", "correct_text", "correct_text") if position == "7" else ("no_action", "no_action", "correct_text")
        fixture["episodes"].append({**copy.deepcopy(slot), "id": slot["episode_id"], "critical_edits": critical,
                                    "current": current_value, "replacement": replacement, "hidden_names": hidden,
                                    "expected_actions": dict(zip(("0", "3", "8"), expected)),
                                    "rationale": "A name has no unique correction without matching role evidence. Only actual visible history identifies the same person; a later correction to that same item takes precedence, and a different role does not transfer. All other visible wording is intentional."})
    return fixture
