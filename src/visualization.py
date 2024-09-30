import matplotlib.pyplot as plt

def plot_sprint_histogram(sprints):
    durations = [s['end'] - s['start'] for s in sprints]
    plt.hist(durations, bins=10)
    plt.xlabel("Sprint Duration (frames)")
    plt.ylabel("Frequency")
    plt.show()
