from coco_agent.benchmark import credential_path, load_credential, save_credential
from coco_agent.cli import show_benchmark_configuration


def test_benchmark_credential_is_external_and_hidden_from_show(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    save_credential("coco_benchmark", "top-secret-value")
    assert load_credential("coco_benchmark") == "top-secret-value"
    assert credential_path().parent == tmp_path / "coco_agent"
    output = []
    show_benchmark_configuration("coco_benchmark", output.append)
    rendered = "\n".join(output)
    assert '"credential_configured": true' in rendered
    assert "top-secret-value" not in rendered
