import glob

import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO


## Main function.


def main():
    ## Get the relevant data files. (Monkey J. 3-rings task.)

    data_dir = "data"
    nwb_extenstion = ".nwb"
    nwb_filenames = glob.glob(f"{data_dir}/*{nwb_extenstion}")

    days_with_3rings_task = ["2016-01-27", "2016-01-28"]
    filenames_with_3rings_task = sorted(
        [
            filename
            for filename in nwb_filenames
            if any(day.replace("-", "") in filename for day in days_with_3rings_task)
        ]
    )

    if filenames_with_3rings_task:
        print("\nFound relevant sessions:\n")
        for filename in filenames_with_3rings_task:
            print(f"\t{filename}")
    else:
        print("No relevant sessions found.")
        return

    ## Read in data from a session.

    first_session = filenames_with_3rings_task[0]
    print(f"\nLoading data from {first_session}...\n")
    with NWBHDF5IO(first_session, "r") as io:
        nwbfile = io.read()

        # Each element is a channel's spiking activity, in the form of a list of times (in seconds from )
        # timestamps that spikes occurred.
        spiking_activity = nwbfile.units["spike_times"][:]
        all_spike_times = np.concatenate(spiking_activity)
        print(max(all_spike_times), min(all_spike_times))


if __name__ == "__main__":
    main()
