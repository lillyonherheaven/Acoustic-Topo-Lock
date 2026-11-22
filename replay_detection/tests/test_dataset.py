import pytest

from replay_detection.dataset_loader import PAArchive


def test_keys_match_audio_and_label_direction(make_archive):
    with PAArchive(make_archive()) as data:
        for partition, index in zip(("train", "dev", "eval"), (1, 2, 3), strict=True):
            first, second, *_ = data.trials[partition]
            assert (first.speaker, first.label, second.label) == (f"PA_{index:04d}", 0, 1)
            assert data.read_audio(first).read(4) == b"fLaC"
            assert data.inspection()["partitions"][partition]["observed_columns"] == [5]


@pytest.mark.parametrize("mode", ["missing", "unlabeled"])
def test_eval_key_unavailable(make_archive, mode):
    with PAArchive(make_archive(eval_mode=mode)) as data:
        assert data.trials["eval"] == []
        assert data.protocols["eval"]["status"] == mode


@pytest.mark.parametrize("options,reason", [
    ({"train_key": "replay"}, "bad ID"), ({"overlap": True}, "speaker-disjoint"),
    ({"missing_audio": True}, "audio missing"),
])
def test_refuse_malformed_partitions(make_archive, options, reason):
    with pytest.raises(ValueError, match=reason), PAArchive(make_archive(**options)):
        pass
