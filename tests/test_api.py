from fastapi.testclient import TestClient

from rl3.api import app


client = TestClient(app)


def test_openenv_endpoints_exist():
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] in {"ok", "healthy"}

    tasks_response = client.get("/tasks")
    assert tasks_response.status_code == 200
    assert len(tasks_response.json()["tasks"]) >= 3

    reset_response = client.post("/reset", json={"task_id": "json_support_ticket", "max_steps": 4})
    assert reset_response.status_code == 200
    payload = reset_response.json()
    assert payload["observation"]["task_id"] == "json_support_ticket"
    assert 0.0 <= payload["reward"] <= 1.0

    step_response = client.post(
        "/step",
        json={
            "action": {
                "new_prompt": "Return valid JSON only using double quotes. Include exactly these keys: product, issue, urgency."
            }
        },
    )
    assert step_response.status_code == 200
    assert 0.0 <= step_response.json()["reward"] <= 1.0

    state_response = client.get("/state")
    assert state_response.status_code == 200

    schema_response = client.get("/schema")
    assert schema_response.status_code == 200
