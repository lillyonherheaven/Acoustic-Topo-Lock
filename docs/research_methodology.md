# Research methodology

## Question and hypotheses

Does adding acoustic context and replay evidence improve acceptance of legitimate voice trials and rejection of replay trials when compared on the same held-out attempts? Hypotheses are that speech retains measurable room-dependent cues, replay affects acoustic features, and combining independently measured evidence reduces replay acceptance under controlled conditions. The null hypothesis is that those added scores do not improve the held-out comparison at the chosen operating point.

## Controlled study

The original plan expected consented human participants, sessions in four rooms, and comparisons across source type and authorized room. The software request identifies 20 planned participants whose recordings are currently unavailable. The supplied blueprint describes a smaller illustrative sample-size design. A future study should define the participant and session split, authorized-room expectations, devices, and capture conditions before collecting trials. Analyze direct speech in authorized and unauthorized rooms and prerecorded playback in authorized and unauthorized rooms as separate conditions.

## Public data stage

Use the official ASVspoof 2019 PA benchmark for replay classification and the pretrained SpeechBrain model for speaker experiments that have suitable consented enrollment and probe trials. Room verification needs its own room-labeled dataset. Public room impulse responses can support controlled DSP studies into reverberation. Proof of incremental benefit requires measured, paired trials with all inputs relevant to each system variant.

## Threat model

The software models speaker mismatch, room mismatch, and playback evidence in uploaded recordings. A playback in a matching room can yield compatible room evidence. A file upload has no trusted binding to the time, device, location, or presence of a person. DSP features can reflect the microphone and gain as well as the room. Document the channel, capture setup, metrics, trial budget, exclusions, and research limitations before interpreting results.
