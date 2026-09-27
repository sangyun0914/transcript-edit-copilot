"use strict";
const $ = (id) => document.getElementById(id);
const names = {
  no_action: "Keep unchanged",
  correct_text: "Correct transcript",
  delete_span: "Remove speech",
  trim_silence: "Shorten gap",
  remove_silence: "Remove gap",
  split_paragraph: "Split caption",
  merge_paragraph: "Merge captions",
};
let batch,
  current,
  index = 0,
  exposure,
  draft,
  shownEdit,
  selected = [],
  rangeStart = null;
let loadId = 0,
  previewId = 0,
  afterUrl,
  busy = false;

async function api(path, body) {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Review-Request": "1",
          },
          body: JSON.stringify(body),
        }
  );
  if (!response.ok) {
    const error = await response.json();
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : "Check the selected action and its parameters."
    );
  }
  return response;
}
const json = async (path, body) => (await api(path, body)).json();
function fail(error) {
  $("error").textContent = error.message;
  $("error").hidden = false;
}
function clearError() {
  $("error").hidden = true;
}
function time(seconds) {
  return seconds == null
    ? "—"
    : `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(
        2,
        "0"
      )}`;
}
function element(tag, text, className) {
  const el = document.createElement(tag);
  el.textContent = text;
  if (className) el.className = className;
  return el;
}
function audioReset() {
  for (const id of ["before-audio", "after-audio", "source-video"]) {
    $(id).pause();
    $(id).removeAttribute("src");
    $(id).load();
  }
  if (afterUrl) URL.revokeObjectURL(afterUrl);
  afterUrl = null;
}
function renderItems(items, target, original) {
  target.replaceChildren();
  let paragraph,
    caption,
    n = 0;
  for (const item of items) {
    if (item.paragraph !== paragraph) {
      paragraph = item.paragraph;
      caption = element("div", "", "caption");
      caption.append(
        element(
          "span",
          `Caption ${++n} · ${item.speaker || "speaker unknown"} · ${(
            items
              .filter((part) => part.paragraph === paragraph)
              .reduce((total, part) => total + part.duration_ms, 0) / 1000
          ).toFixed(1)}s shown`,
          "caption-label"
        )
      );
      target.append(caption);
    }
    const token = element(
      original ? "button" : "span",
      item.type === "text"
        ? item.text
        : `[${(item.duration_ms / 1000).toFixed(2)}s gap]`,
      "token"
    );
    token.dataset.itemId = String(item.id);
    if (original) {
      token.type = "button";
      token.addEventListener("click", () => selectTarget(item.id));
    }
    if (item.type !== "text") token.classList.add("gap");
    if (shownEdit?.target?.includes(item.id)) {
      if (
        original &&
        ["delete_span", "remove_silence"].includes(shownEdit.action)
      )
        token.classList.add("removed");
      else token.classList.add("changed");
    }
    if (original && !$("adjustment").hidden && selected.includes(item.id))
      token.classList.add("target");
    caption.append(token, document.createTextNode(" "));
  }
}
function renderNavigation() {
  const count = batch.cases.filter(
    (c) => c.verdict && c.verdict !== "clear"
  ).length;
  $("progress").textContent = `${count} of ${batch.cases.length} labeled`;
  $("progress-bar").max = batch.cases.length;
  $("progress-bar").value = count;
  $("case-list").replaceChildren();
  batch.cases.forEach((row, i) => {
    const button = element("button", "");
    button.append(
      element(
        "span",
        `${String(i + 1).padStart(2, "0")} · ${time(row.position_s)}`
      )
    );
    button.append(
      element(
        "span",
        row.verdict && row.verdict !== "clear" ? row.verdict : "To review",
        "case-state"
      )
    );
    button.classList.toggle("active", i === index);
    button.classList.toggle(
      "reviewed",
      !!row.verdict && row.verdict !== "clear"
    );
    button.setAttribute("aria-current", i === index ? "true" : "false");
    button.addEventListener("click", () => loadCase(i).catch(fail));
    $("case-list").append(button);
  });
  $("previous").disabled = index === 0;
  $("next").disabled = index === batch.cases.length - 1;
}
async function loadCase(next) {
  if (busy || next < 0 || next >= batch.cases.length) return;
  const serial = ++loadId;
  ++previewId;
  clearError();
  audioReset();
  index = next;
  current = null;
  draft = null;
  shownEdit = null;
  exposure = null;
  $("review-form").hidden = true;
  $("draft-details").hidden = true;
  $("adjustment").hidden = true;
  $("selection-help").hidden = true;
  $("before").classList.remove("selecting");
  $("after").replaceChildren();
  $("action-badge").textContent = "";
  $("hidden-draft").hidden = false;
  $("after-audio").hidden = true;
  $("after-title").textContent = "Proposed edit";
  for (const id of ["notes", "initial-note", "initial-judgment", "meaning"])
    $(id).value = "";
  $("audio-quality").value = "not_checked";
  $("show-now").value = "unsure";
  renderNavigation();
  const data = await json(`/api/cases/${batch.cases[index].id}`);
  if (serial !== loadId) return;
  current = data;
  localStorage.setItem("nep-review-position", current.id);
  $("case-number").textContent = `CASE ${index + 1} / ${batch.cases.length}`;
  $("case-title").textContent = `Review the excerpt at ${time(
    current.position_s
  )}`;
  const origin = {
    constructed_corruption: "Contains a controlled spelling change",
    constructed_caption_boundary: "Contains an inserted caption boundary",
    constructed_source_window: "Constructed context from the recording",
  };
  $("context-note").textContent = `${
    origin[current.context_origin] || current.context_origin
  } · ${
    current.has_observed_action
      ? "Observed editing action available for review"
      : "No observed next action"
  } · ${
    current.input.items.every((item) => item.boundary_status === "verified")
      ? "Timing marked verified"
      : "ASR timing unverified"
  }${
    current.previously_exposed
      ? " · Draft previously viewed; this case cannot receive a fresh blind label"
      : ""
  }.`;
  renderItems(current.input.items, $("before"), true);
  $("before-audio").hidden = !current.has_audio;
  if (current.has_audio)
    $("before-audio").src = `/api/cases/${current.id}/audio`;
  $("video-context").hidden = !current.video;
  if (current.video) {
    const video = $("source-video");
    video.onloadedmetadata = () => {
      if (serial === loadId) video.currentTime = current.video.start_s;
    };
    video.src = `/api/cases/${current.id}/video`;
  }
  $("save-status").textContent = current.reviewed
    ? "Previously labeled · history below"
    : "Not labeled";
  $("independent").hidden = $("mode").value !== "blind";
  if ($("mode").value === "assisted") await reveal();
}
async function reveal() {
  if (!current) return;
  const id = current.id,
    serial = loadId;
  const result = await json(`/api/cases/${id}/reveal`, {
    mode: $("mode").value,
    initial_judgment: $("initial-judgment").value || null,
    initial_note: $("initial-note").value,
  });
  if (serial !== loadId) return;
  exposure = result.exposure;
  draft = result.proposal;
  $("independent").hidden = true;
  $("hidden-draft").hidden = true;
  $("review-form").hidden = false;
  $("draft-details").hidden = false;
  $("rationale").textContent = result.rationale;
  $("history").textContent = current.input.recent_edits.length
    ? JSON.stringify(current.input.recent_edits)
    : "No editing history was recorded for this calibration excerpt. Do not assume an editing preference.";
  await updatePreview(draft);
  if (serial === loadId) await history();
}
async function updatePreview(edit) {
  const serial = ++previewId,
    caseSerial = loadId,
    id = current.id;
  const result = await json(`/api/cases/${id}/preview`, edit);
  let url;
  if (current.has_audio)
    url = URL.createObjectURL(
      await (await api(`/api/cases/${id}/audio`, edit)).blob()
    );
  if (serial !== previewId || caseSerial !== loadId) {
    if (url) URL.revokeObjectURL(url);
    return;
  }
  shownEdit = structuredClone(edit);
  renderItems(current.input.items, $("before"), true);
  renderItems(result.items, $("after"), false);
  $("action-badge").textContent = names[edit.action];
  $("warnings").replaceChildren(...result.warnings.map((w) => element("p", w)));
  $("warnings").hidden = !result.warnings.length;
  $("after-audio").pause();
  if (afterUrl) URL.revokeObjectURL(afterUrl);
  afterUrl = url;
  $("after-audio").hidden = !url;
  if (url) $("after-audio").src = url;
  $("audio-quality").value = [
    "delete_span",
    "trim_silence",
    "remove_silence",
  ].includes(edit.action)
    ? "not_checked"
    : "not_applicable";
}
function selectTarget(id) {
  if ($("adjustment").hidden) return;
  if (rangeStart === null) {
    rangeStart = id;
    selected = [id];
  } else {
    const ids = current.input.items.map((i) => i.id);
    const a = ids.indexOf(rangeStart),
      b = ids.indexOf(id);
    selected = ids.slice(Math.min(a, b), Math.max(a, b) + 1);
    rangeStart = null;
  }
  selectedText();
  renderItems(current.input.items, $("before"), true);
}
function selectedText() {
  $("selected-target").textContent = `Selected: ${
    current.input.items
      .filter((i) => selected.includes(i.id))
      .map((i) => i.text || `[${i.duration_ms}ms gap]`)
      .join(" ") || "choose items in Original"
  }`;
}
function adjustmentFields() {
  const action = $("adjust-action").value;
  $("replacement-label").hidden = action !== "correct_text";
  $("duration-label").hidden = action !== "trim_silence";
}
function adjusted() {
  const action = $("adjust-action").value;
  const edit = { action };
  if (action !== "no_action") edit.target = selected;
  if (action === "correct_text") edit.text = $("replacement").value;
  if (action === "trim_silence") edit.keep_ms = Number($("duration").value);
  return edit;
}
function playedThrough(id) {
  const audio = $(id);
  let played = 0;
  for (let i = 0; i < audio.played.length; i++)
    played += audio.played.end(i) - audio.played.start(i);
  return (
    Number.isFinite(audio.duration) &&
    audio.duration > 0 &&
    played >= audio.duration * 0.9
  );
}
async function save(verdict) {
  if (!current || !exposure || busy) return;
  clearError();
  if (
    verdict === "acceptable" &&
    JSON.stringify(shownEdit) !== JSON.stringify(draft)
  )
    throw new Error(
      "Use Save adjustment for the edited preview, or return to the original draft."
    );
  if (
    verdict === "adjust" &&
    JSON.stringify(shownEdit) !== JSON.stringify(adjusted())
  )
    throw new Error("Preview the current adjustment before saving it.");
  busy = true;
  $("save-status").textContent = "Saving…";
  try {
    const label = await json(`/api/cases/${current.id}/labels`, {
      case_hash: current.case_hash,
      base_revision: current.revision,
      exposure_id: exposure.id,
      verdict,
      chosen_edit: verdict === "adjust" ? adjusted() : null,
      meaning: $("meaning").value || null,
      audio: $("audio-quality").value,
      show_now: $("show-now").value,
      notes: $("notes").value,
      listened_before: playedThrough("before-audio"),
      listened_after: playedThrough("after-audio"),
    });
    current.revision = label.revision;
    batch.cases[index].verdict = label.verdict;
    $("save-status").textContent =
      verdict === "clear"
        ? "Label cleared · history preserved"
        : "Saved locally";
    renderNavigation();
    await history();
  } catch (error) {
    $("save-status").textContent = "Not saved";
    throw error;
  } finally {
    busy = false;
  }
}
async function history() {
  const id = current.id,
    serial = loadId;
  const rows = await json(`/api/cases/${id}/labels`);
  if (serial !== loadId) return;
  $("saved-labels").replaceChildren(
    ...rows
      .slice(-5)
      .reverse()
      .map((row) =>
        element(
          "div",
          `${new Date(row.at).toLocaleString()} · ${row.verdict}${
            row.chosen_edit ? ` · ${names[row.chosen_edit.action]}` : ""
          }${row.notes ? ` — ${row.notes}` : ""}`,
          "history-entry"
        )
      )
  );
  if (!rows.length) $("saved-labels").textContent = "No saved labels yet.";
}
const on = (id, callback) =>
  $(id).addEventListener("click", () =>
    Promise.resolve().then(callback).catch(fail)
  );
on("previous", () => loadCase(index - 1));
on("next", () => loadCase(index + 1));
on("reveal", reveal);
on("accept", () => save("acceptable"));
on("reject", () => save("reject"));
on("keep", () => save("no_action"));
on("unsure", () => save("unsure"));
on("clear-label", () => save("clear"));
on("adjust-toggle", () => {
  $("adjustment").hidden = false;
  $("selection-help").hidden = false;
  $("before").classList.add("selecting");
  selected = [...(shownEdit.target || [])];
  rangeStart = null;
  $("adjust-action").value = shownEdit.action;
  $("replacement").value = shownEdit.text || "";
  $("duration").value = shownEdit.keep_ms || 300;
  adjustmentFields();
  selectedText();
  renderItems(current.input.items, $("before"), true);
});
on("preview-adjustment", () => updatePreview(adjusted()));
on("save-adjustment", () => save("adjust"));
on("cancel-adjustment", async () => {
  $("adjustment").hidden = true;
  $("selection-help").hidden = true;
  $("before").classList.remove("selecting");
  await updatePreview(draft);
});
$("adjust-action").addEventListener("change", adjustmentFields);
$("mode").addEventListener("change", () => {
  localStorage.setItem("nep-review-mode", $("mode").value);
  loadCase(index).catch(fail);
});
on("video-return", () => {
  if (current?.video) $("source-video").currentTime = current.video.start_s;
});
const players = ["before-audio", "after-audio", "source-video"];
for (const id of players)
  $(id).addEventListener("play", () => {
    for (const other of players) if (other !== id) $(other).pause();
  });
document.addEventListener("keydown", (e) => {
  if (
    e.target.closest("input,textarea,select,audio,video") ||
    e.metaKey ||
    e.ctrlKey ||
    e.altKey
  )
    return;
  const key = {
    a: "accept",
    r: "reject",
    u: "unsure",
    ArrowLeft: "previous",
    ArrowRight: "next",
  }[e.key];
  if (key && !$(key).closest("[hidden]")) {
    e.preventDefault();
    $(key).click();
  }
});
(async () => {
  batch = await json("/api/batch");
  $("batch-title").textContent = batch.title;
  $("batch-note").textContent = batch.note;
  $("mode").value =
    localStorage.getItem("nep-review-mode") === "blind" ? "blind" : "assisted";
  const saved = batch.cases.findIndex(
    (c) => c.id === localStorage.getItem("nep-review-position")
  );
  await loadCase(saved >= 0 ? saved : 0);
})().catch(fail);
