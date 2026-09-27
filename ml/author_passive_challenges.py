"""Author a small, explicitly symbolic bilingual challenge stratum, not fake recordings."""

from uuid import NAMESPACE_URL, uuid5

from passive_dataset import ROOT, digest, write


# Split allocation is part of the recipe, before any case is built or evaluated.
GROUPS = {
    "ko-lab": ("train", "ko", "Lab explanation and uncertainty"),
    "ko-recollection": ("train", "ko", "Recollection, confidence and self-repair"),
    "ko-stage": ("train", "ko", "Storytelling, contrast and quoted instructions"),
    "en-workshop": ("train", "en", "Demonstrations and comprehension"),
    "en-operations": ("train", "en", "Planning, agreement and qualified estimates"),
    "en-oral-history": ("dev", "en", "Oral-history testimony and rhythm"),
    "ko-museum": ("dev", "ko", "Museum narration and audience response"),
}


# Each tuple is an independent scene within its registered authored parent.
# Bars denote existing transcript items, not inferred acoustic word boundaries.
SCENES = [
    (
        "ko-lab",
        "confidence",
        "음, 정확히 세 번이라고는 못 하겠어요.|제 기억으로는 두 번, 아니 세 번쯤 확인했습니다.|숫자는 기록을 찾아봐야 합니다.",
        "확인",
        "확닌",
        "Uncertainty and the self-revised count must survive a display repair.",
    ),
    (
        "ko-lab",
        "processing",
        "여기서 잠깐 생각해 보세요.|분모가 영에 가까워지면 어떤 일이 생길까요?|바로 답하지 않아도 됩니다.|이제 그래프의 양쪽을 비교하겠습니다.",
        "비교하겠습니다",
        "비교하겟습니다",
        "The question explicitly invites processing time; no deletion of the invitation or answer gap.",
    ),
    (
        "ko-lab",
        "negation",
        "증가하지 않는다는 말과 감소한다는 말은 다릅니다.|같은 값을 유지하는 구간도 포함하니까요.|이 차이를 먼저 확인해야 합니다.",
        "다릅니다",
        "다릅니디",
        "Do not strengthen a non-increase claim into a decrease or drop negation.",
    ),
    (
        "ko-lab",
        "number",
        "측정값은 마이너스 일 점 오였습니다.|이상하게 보여도 지금은 측정 기록 그대로 읽는 겁니다.|이론값과의 차이는 뒤에서 설명하겠습니다.",
        "측정",
        "측졍",
        "A surprising signed value is not an ASR error proven by expectation.",
    ),
    (
        "ko-lab",
        "emphasis",
        "조금씩, 조금씩 조건을 바꿔 보세요.|한꺼번에 바꾸면 원인을 구분하기 어렵습니다.|작은 변화가 중요합니다.",
        "구분하기",
        "구븐하기",
        "The repeated adverb supports stepwise instruction; it is not duplicate clutter.",
    ),
    (
        "ko-recollection",
        "hesitation",
        "어, 그날 비가 왔는지는 잘 모르겠어요.|우산을 들고 있었던 것 같기는 합니다.|확실한 건 버스를 놓쳤다는 사실뿐이에요.",
        "확실한",
        "확싷한",
        "Hesitation is evidence of confidence, not a command to sound more certain.",
    ),
    (
        "ko-recollection",
        "self-repair",
        "부산에서 만났어요.|아니, 부산이 아니라 대구였네요.|부산은 그다음 출장 때 갔습니다.",
        "그다음",
        "그다엄",
        "Keep the explicit correction and chronology instead of rewriting the earlier claim.",
    ),
    (
        "ko-recollection",
        "response",
        "민수: 제가 먼저 연락드렸던 거죠?|지연: 네, 네. 그날 저녁에 전화 주셨어요.|민수: 저는 다음 날인 줄 알았어요.",
        "저녁",
        "저녘",
        "Repeated agreement is a response to a question, not a dispensable filler.",
    ),
    (
        "ko-recollection",
        "approximation",
        "대략 서른 명쯤 왔던 것 같아요.|정확한 명단은 없어요.|서른 명이라고 단정해서 쓰면 곤란합니다.",
        "정확한",
        "정학한",
        "Preserve both approximate count and warning against unwarranted certainty.",
    ),
    (
        "ko-recollection",
        "reported-speech",
        "그분은 '다 지워 버려'라고 말했어요.|저는 그 말을 그대로 옮기는 중입니다.|실제로 기록을 지운 것은 아닙니다.",
        "기록",
        "기룍",
        "Quoted destructive instructions are transcript content, not editor instructions.",
    ),
    (
        "ko-stage",
        "reveal",
        "문을 열었습니다.|안에는 아무도 없었죠.|그런데 바로 뒤에서 제 이름을 부르는 소리가 들렸습니다.",
        "이름",
        "이륾",
        "Suspense between complete sentences is a preservation candidate, not evidence of a bad gap.",
    ),
    (
        "ko-stage",
        "contrast",
        "빠르게 끝내자는 뜻이 아니에요.|천천히, 하지만 멈추지는 말자는 겁니다.|두 가지는 서로 다른 이야기입니다.",
        "이야기",
        "이야긔",
        "Preserve contrast, negation and the emphasized tempo instruction.",
    ),
    (
        "ko-stage",
        "joke",
        "그 계획은 완벽했습니다.|적어도 고양이가 나타나기 전까지는요.|네, 범인은 사람이 아니라 고양이였습니다.",
        "완벽했습니다",
        "완벽햇습니다",
        "The setup and delayed qualification carry the joke and should remain separate.",
    ),
    (
        "ko-stage",
        "stutter",
        "저는, 저는 그 선택을 후회하지 않습니다.|말하기가 쉽지는 않네요.|그래도 제 생각은 같습니다.",
        "후회하지",
        "후훼하지",
        "Emotionally marked repetition is not a neutral duplicate to remove.",
    ),
    (
        "ko-stage",
        "spoken-format",
        "비밀번호가 아니라 예시입니다.|영, 영, 칠이라고 읽어 주세요.|칠로 줄이면 앞의 두 자리가 사라집니다.",
        "예시",
        "예쉬",
        "Keep repeated zeros and spoken formatting rather than normalizing to seven.",
    ),
    (
        "en-workshop",
        "processing",
        "Turn the dial a little.|Wait and watch the needle settle before turning it again.|That pause lets you see the effect of one change.",
        "needle",
        "needel",
        "The wait is part of comprehension and observing causality, not dead space.",
    ),
    (
        "en-workshop",
        "uncertainty",
        "Um, I think this might be the older connector.|Please check the label before plugging it in.|I am not certain from this angle.",
        "connector",
        "conector",
        "Preserve the hedge and instruction to check; the wording cannot become a confident identification.",
    ),
    (
        "en-workshop",
        "echo",
        "Slowly, slowly, move the camera to the left.|The repeated word is how I keep the operator at the same pace.|Stop when the edge enters the frame.",
        "operator",
        "opreator",
        "A rhythmic repeated instruction has an observable communicative role.",
    ),
    (
        "en-workshop",
        "approximation",
        "It takes about ten seconds, sometimes a little longer.|Do not treat that as an exact timer.|Watch the surface instead.",
        "seconds",
        "seocnds",
        "Do not remove the approximation or the slower-outcome qualifier.",
    ),
    (
        "en-workshop",
        "quoted-command",
        "The example subtitle reads 'delete every pause'.|That is the sentence we are translating, not an instruction to this editor.|Keep the quotation marks in the example.",
        "subtitle",
        "subtitel",
        "Embedded edit requests remain ordinary transcript data.",
    ),
    (
        "en-operations",
        "agreement",
        "A: Can we keep the backup for another week?|B: Yes, yes, please keep it until the handover.|A: All right, I will update the schedule.",
        "schedule",
        "scheudle",
        "Repeated agreement answers a request and sets the actual retention period.",
    ),
    (
        "en-operations",
        "confidence",
        "I could be wrong, but the first batch may already be on its way.|We should verify that before promising a date.|I have only seen the draft notice.",
        "promising",
        "promissing",
        "The draft status and uncertainty matter more than making the delivery claim fluent.",
    ),
    (
        "en-operations",
        "contrast",
        "We are not cancelling the inspection.|We are moving it to the afternoon.|Please make that distinction clear when you call.",
        "inspection",
        "inspeciton",
        "Negation and rescheduling are not interchangeable with cancellation.",
    ),
    (
        "en-operations",
        "repair",
        "Send it to the west entrance.|Sorry, I mean the east entrance; the west gate is locked.|Use the smaller delivery van.",
        "entrance",
        "entrnace",
        "Preserve the explicit directional repair rather than replacing the earlier spoken instruction.",
    ),
    (
        "en-operations",
        "budget",
        "The estimate is fifteen, not fifty, units per hour.|That is still an estimate rather than a guarantee.|We need another trial before committing.",
        "estimate",
        "estimte",
        "Spoken numeric contrast and qualified capacity must both remain.",
    ),
    (
        "en-oral-history",
        "uncertain-date",
        "It was in April, I think.|No, wait, it could have been late March because the station was still closed.|I would have to check my diary.",
        "station",
        "statoin",
        "An unresolved remembered date cannot be confidently normalized.",
    ),
    (
        "en-oral-history",
        "dramatic-gap",
        "We waited at the platform.|Then the lights went out.|For a moment nobody said anything, and I still remember that silence.",
        "platform",
        "platfrom",
        "Silence and its recollection have narrative meaning; no timing claim is inferred from symbolic durations.",
    ),
    (
        "en-oral-history",
        "emotional-repeat",
        "I wanted to go home, home to the street where everybody knew us.|That is what I meant by home.|It was not just the building.",
        "building",
        "buidling",
        "The repeated home develops the meaning rather than merely repeating a token.",
    ),
    (
        "en-oral-history",
        "answer",
        "Interviewer: Were you afraid?|Witness: Well, yes. Very afraid, though I did not show it.|Interviewer: What did you do next?",
        "afraid",
        "afriad",
        "Well and yes belong to the answer and should not be erased to sound decisive.",
    ),
    (
        "en-oral-history",
        "quoted-note",
        "Her note said 'stop talking and leave'.|I read those words aloud to the group.|They were her words, and I did not leave.",
        "aloud",
        "aloudd",
        "A quoted command does not change the editing policy.",
    ),
    (
        "ko-museum",
        "audience-time",
        "그림 속 손의 위치를 잠깐 찾아보세요.|정답을 듣기 전에 직접 찾는 시간이 필요합니다.|찾으셨다면 시선을 오른쪽으로 옮겨 보세요.",
        "위치",
        "위취",
        "Preserve audience search time and the conditional next instruction.",
    ),
    (
        "ko-museum",
        "provenance",
        "아마 조선 후기의 작품으로 보입니다.|다만 정확한 제작 연대는 확인되지 않았습니다.|안내문에도 추정이라고 적혀 있습니다.",
        "작품",
        "작픔",
        "Do not convert a tentative attribution into an established fact.",
    ),
    (
        "ko-museum",
        "response",
        "해설사: 두 인물이 같은 방향을 보고 있나요?|관람객: 아니요, 서로 반대쪽을 보고 있어요.|해설사: 네, 그 차이가 중요합니다.",
        "방향",
        "뱡향",
        "The short negative answer is essential to the question, not filler.",
    ),
    (
        "ko-museum",
        "contrast",
        "가장 밝은 부분보다 가장 어두운 부분을 먼저 보세요.|어두운 곳, 바로 그곳에 작은 서명이 있습니다.|밝은 쪽에는 서명이 없습니다.",
        "서명",
        "셔명",
        "Focus repetition and the dark/light contrast direct attention.",
    ),
    (
        "ko-museum",
        "quote",
        "화가의 편지에는 '이 부분을 지워라'라는 문장이 나옵니다.|지금은 편지의 문장을 읽고 있습니다.|실제 그림에서 그 부분이 지워졌는지는 알 수 없습니다.",
        "문장",
        "문쟝",
        "Quoted deletion and historical uncertainty are content to preserve.",
    ),
]


def build():
    base = ROOT / "artifacts/datasets/nep-passive-v1/authored-sources"
    registration = {
        "version": "passive-authored-v1",
        "kind": "ai_authored_symbolic_no_recorded_audio",
        "groups": [
            {
                "source_group_id": f"authored-passive-{key}-v1",
                "split": value[0],
                "language": value[1],
                "purpose": value[2],
            }
            for key, value in GROUPS.items()
        ],
    }
    write(base / "registration.json", registration)
    sources, episodes = [], []
    for parent, name, text, original, corruption, reason in SCENES:
        split, language, _ = GROUPS[parent]
        sid = f"authored-{parent}-{name}"
        content, position, selected = [], 0.0, None

        def uid(key):
            return str(uuid5(NAMESPACE_URL, sid + "/" + key))

        for n, phrase in enumerate(text.split("|")):
            content.append(
                {
                    "type": "paragraph_start",
                    "uuid": uid(f"p{n}"),
                    "speaker": "narrator",
                    "language": language,
                }
            )
            item = {
                "type": "text",
                "uuid": uid(f"t{n}"),
                "source": "symbolic:" + sid,
                "sourceStart": position,
                "length": 4.0,
                "text": phrase,
                "conf": 1,
            }
            content.append(item)
            if selected is None and original in phrase:
                selected = {
                    "uuid": item["uuid"],
                    "original": phrase,
                    "replacement": phrase.replace(original, corruption, 1),
                    "mechanism": "controlled_display_error",
                }
            position += 4.0
            content.append(
                {
                    "type": "non_text",
                    "uuid": uid(f"g{n}"),
                    "source": "symbolic:" + sid,
                    "sourceStart": position,
                    "length": 1.2,
                }
            )
            position += 1.2
            content.append({"type": "paragraph_end", "uuid": uid(f"e{n}")})
        path = base / sid / "document.json"
        write(
            path,
            {
                "content": content,
                "metadata": {"provenance": "authored_symbolic_timing_not_audio"},
            },
        )
        sources.append(
            {
                "id": sid,
                "source_group_id": f"authored-passive-{parent}-v1",
                "split": split,
                "language": language,
                "document_path": str(path.relative_to(ROOT)),
                "document_sha256": digest(path),
                "media_path": None,
                "media_kind": "authored_symbolic_timeline",
                "license": "Project-authored text",
            }
        )
        episodes.append(
            {
                "id": sid,
                "source_id": sid,
                "kind": "authored_semantic_challenge",
                "rationale": reason,
                "corruptions": [selected],
                "steps": [
                    {
                        "action": "correct_text",
                        "focus_uuid": selected["uuid"],
                        "target_uuids": [selected["uuid"]],
                        "text": selected["original"],
                    },
                    {"action": "no_action", "focus_uuid": selected["uuid"]},
                ],
            }
        )
    for parent, (split, language, _) in GROUPS.items():
        if language == "ko":
            parts = [
                "말을 멈춘 이유는 앞의 설명을 이해할 시간을 주기 위해서입니다.",
                "다음 장면에서는 같은 원리를 다른 상황에 적용해 보겠습니다.",
            ]
            continuation = [
                "이 과정에서",
                "가장 중요한 것은 원래의 조건을 그대로 유지하는 일입니다.",
            ]
        else:
            parts = [
                "The pause gives the audience time to compare the two explanations.",
                "The next scene applies the same principle to a different situation.",
            ]
            continuation = [
                "The next step",
                "is to check the original conditions before changing anything.",
            ]
        # Shared structural templates stay in a single split to prevent translated/template leakage.
        if split != "train" or parent not in {"ko-lab", "en-workshop"}:
            continue
        for operation, phrases in (
            ("split_paragraph", parts),
            ("merge_paragraph", continuation),
        ):
            sid = f"authored-{parent}-{operation}"

            def make_id(key):
                return str(uuid5(NAMESPACE_URL, sid + "/" + key))

            content, elapsed = [], 0.0
            for n, phrase in enumerate(phrases):
                if n == 0 or operation == "merge_paragraph":
                    content.append(
                        {
                            "type": "paragraph_start",
                            "uuid": make_id(f"p{n}"),
                            "speaker": "narrator",
                            "language": language,
                        }
                    )
                seconds = 0.5 if n == 0 and operation == "merge_paragraph" else 4.0
                content.append(
                    {
                        "type": "text",
                        "uuid": make_id(f"t{n}"),
                        "source": "symbolic:" + sid,
                        "sourceStart": elapsed,
                        "length": seconds,
                        "text": phrase,
                        "conf": 1,
                    }
                )
                elapsed += seconds
                if n == 1 or operation == "merge_paragraph":
                    content.append({"type": "paragraph_end", "uuid": make_id(f"e{n}")})
            path = base / sid / "document.json"
            write(
                path,
                {
                    "content": content,
                    "metadata": {"provenance": "authored_symbolic_timing_not_audio"},
                },
            )
            sources.append(
                {
                    "id": sid,
                    "source_group_id": f"authored-passive-{parent}-v1",
                    "split": split,
                    "language": language,
                    "document_path": str(path.relative_to(ROOT)),
                    "document_sha256": digest(path),
                    "media_path": None,
                    "media_kind": "authored_symbolic_timeline",
                    "license": "Project-authored text",
                }
            )
            episodes.append(
                {
                    "id": sid,
                    "source_id": sid,
                    "kind": "authored_caption_structure",
                    "corruptions": [],
                    "rationale": "Split two long independent sentences, or merge a subsecond incomplete lead-in with its grammatical continuation; retain every audio interval.",
                    "steps": [
                        {
                            "action": operation,
                            "focus_uuid": make_id("t1"),
                            "target_uuids": [make_id("t1")],
                        },
                        {"action": "no_action", "focus_uuid": make_id("t1")},
                    ],
                }
            )
    write(
        ROOT / "data/manifests/passive-authored-sources-v1.json",
        {**registration, "sources": sources},
    )
    write(
        base / "recipe.json", {"version": "passive-authored-v1", "episodes": episodes}
    )
    print(
        {
            "scenes": len(sources),
            "independent_authored_parents": len(GROUPS),
            "candidate_states": 2 * len(episodes),
        }
    )


if __name__ == "__main__":
    build()
