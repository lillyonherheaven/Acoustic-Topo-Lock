# Dataset manifests

Obtain authorized copies of the data and inspect the archive and labeled protocol files manually. Put the manifest at `data/metadata/replay.json` or `data/metadata/room.json`. Relative protocol and audio directory paths resolve from the manifest directory. The Dataset Information page displays the actual checked file count when you inspect the manifest.

Use the shape below and replace all values with inspected information. The named paths are examples.

```json
{
  "name": "ASVspoof 2019 PA",
  "version": "archive release or checksum",
  "source": "https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45",
  "license": "license from the authorized release",
  "download_date": "YYYY-MM-DD",
  "partitions": {
    "train": {"protocol": "../raw/asvspoof2019_pa/train-key.txt", "audio_dir": "../raw/asvspoof2019_pa/train-audio"},
    "dev": {"protocol": "../raw/asvspoof2019_pa/dev-key.txt", "audio_dir": "../raw/asvspoof2019_pa/dev-audio"},
    "eval": {"protocol": "../raw/asvspoof2019_pa/eval-key.txt", "audio_dir": "../raw/asvspoof2019_pa/eval-audio"}
  }
}
```

For replay, supply the labeled five-field PA countermeasure protocol for each official split. The parser reads speaker ID, recording ID, environment, attack condition, and key. It maps `bonafide` to `live` and `spoof` to `replay`. An unlabeled evaluation trial list must be paired with the official labeled key before use. Keep the official train, dev, and eval assignments. [ASVspoof 2019](https://www.asvspoof.org/index2019.html)

For room data, use one CSV per partition with the header `recording_id,participant_id,session_id,room,audio_path`. Paths in the CSV are relative to that partition's declared audio directory. Split by participant before creating the CSVs and reserve separate enrollment recordings for verification trials.

Inspection verifies path existence, known labels, and IDs, paths, and participant disjointness. Feature extraction decodes each selected recording and reports the recording ID if validation fails. Audit original asset lineage, duplicate audio content under different IDs, device assignments, and exclusions when preparing the protocol. Preprocessing version, label mapping, and counts are derived from the inspected manifest and local files.
