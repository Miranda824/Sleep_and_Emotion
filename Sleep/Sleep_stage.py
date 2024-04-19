


import matplotlib.pyplot as plt
import mne

# Load raw data and annotations
raw_train = mne.io.read_raw_edf("datasets/sleep-edfx/SC4112E0-PSG.edf", preload=True)

annot_train = mne.read_annotations("datasets/sleep-edfx/SC4112EC-Hypnogram.edf")
print(annot_train)



# Crop annotations

annot_train.crop(0, 60000)
raw_train.set_annotations(annot_train, emit_warning=False)

# Sleep stage event mapping
event_id = {
    'Sleep stage W': 1,
    'Sleep stage 1': 2,
    'Sleep stage 2': 3,
    'Sleep stage 3': 4,
    'Sleep stage R': 5
}

# Extract events from annotations
events_train, _ = mne.events_from_annotations(raw_train, event_id=event_id, chunk_duration=30.)

# Plot events
mne.viz.plot_events(events_train, event_id=event_id, sfreq=raw_train.info['sfreq'])

# Save the color codes for further plotting
stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
