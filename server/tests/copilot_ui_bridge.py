"""JSON-lines bridge from the Electron test to the real gateway and a fake model."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.copilot import (  # noqa: E402
    Completion,
    CopilotService,
    contract,
    create_copilot_router,
)


class FixtureAdapter:
    model = {"id": "synthetic-ui-fixture", "sha256": "0" * 64}
    contract_version = "passive-nep-actions-v1"

    def __init__(self):
        self.action = None

    def health(self):
        return {"status": "ready", "model": self.model, "message": "Synthetic UI test."}

    def apply_chat_template(self, messages, **kwargs):
        return [1] * 100

    async def generate(self, messages, output_schema, timeout):
        snapshot = contract(self.contract_version).parse_snapshot(
            json.loads(messages[-1]["content"])
        )
        prediction = self.action or {"action": "no_action"}
        if self.action is None:
            corrections = {"recieve": "receive", "데이타": "데이터"}
            for item in snapshot.items:
                if item.text in corrections:
                    prediction = {
                        "action": "correct_text",
                        "target": [item.id],
                        "text": corrections[item.text],
                    }
                    break
        return Completion(json.dumps(prediction, ensure_ascii=False), "stop")


def main():
    adapter = FixtureAdapter()
    app = FastAPI()
    app.state.copilot = CopilotService(adapter)
    app.include_router(create_copilot_router(lambda: None))
    with TestClient(app) as client:
        for line in sys.stdin:
            message = json.loads(line)
            try:
                if message.get("command") == "action":
                    adapter.action = message.get("prediction")
                    result = {"status": 200, "body": {"configured": True}}
                else:
                    response = client.request(
                        message["method"], message["route"], json=message.get("body")
                    )
                    result = {"status": response.status_code, "body": response.json()}
                print(json.dumps({"id": message["id"], **result}, ensure_ascii=False), flush=True)
            except Exception as error:
                print(
                    json.dumps(
                        {
                            "id": message["id"],
                            "status": 500,
                            "body": {"detail": str(error)},
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
