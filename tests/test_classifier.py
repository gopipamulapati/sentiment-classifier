from classifier import build_model, load_csv, main, train_and_eval


def test_accuracy_reasonable():
    texts, labels = load_csv("data/reviews.csv")
    _, acc = train_and_eval(texts, labels)
    assert acc >= 0.7


def test_cli_roundtrip(tmp_path, capsys):
    out = str(tmp_path / "m.pkl")
    main(["train", "--out", out])
    main(["predict", "--model", out, "absolutely love it, great quality"])
    assert capsys.readouterr().out.strip().endswith("positive")
