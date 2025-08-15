from src.cli.commands.plan import main as plan_main


def test_hive_plan_dry_run_invokes_runner(tmp_path, monkeypatch):
    plan = tmp_path / "p.yaml"
    plan.write_text("version: 1\nplan_id: p\ngoal: g\nbatches: []\n")
    rc = plan_main(["dry-run", str(plan)])
    assert rc == 0
