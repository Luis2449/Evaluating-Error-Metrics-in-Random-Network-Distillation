import argparse
import re
import struct
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from tensorboardX.proto.event_pb2 import Event


EVENT_FILE_PATTERN = 'events.out.tfevents.*'


def read_scalars(event_path):
    """Read scalar summaries from one TensorBoard event file."""
    scalars = defaultdict(list)

    with event_path.open('rb') as event_file:
        while True:
            header = event_file.read(12)
            if not header:
                break
            if len(header) != 12:
                raise ValueError('Incomplete TensorBoard record header')

            record_length = struct.unpack('<Q', header[:8])[0]
            event_data = event_file.read(record_length)
            footer = event_file.read(4)
            if len(event_data) != record_length or len(footer) != 4:
                raise ValueError('Incomplete TensorBoard event record')

            event = Event()
            event.ParseFromString(event_data)
            if not event.HasField('summary'):
                continue

            for value in event.summary.value:
                if value.HasField('simple_value'):
                    scalars[value.tag].append(
                        (event.step, value.simple_value, event.wall_time))

    return scalars


def safe_filename(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')


def plot_scalar(tag, values, output_path):
    """Save a static plot for one logged training metric."""
    steps = [value[0] for value in values]
    scalar_values = [value[1] for value in values]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, scalar_values, linewidth=1)
    ax.set_title(tag)
    ax.set_xlabel('TensorBoard Step')
    ax.set_ylabel(tag.split('/')[-1].replace('_', ' ').title())
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def merge_scalars(event_paths):
    """Merge event files from one run directory in TensorBoard step order."""
    merged = defaultdict(list)
    for event_path in event_paths:
        for tag, values in read_scalars(event_path).items():
            merged[tag].extend(values)

    return {
        tag: sorted(values, key=lambda value: (value[0], value[2]))
        for tag, values in merged.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate one diagram for every scalar in TensorBoard event files.')
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=Path('models'),
        help='Directory recursively searched for TensorBoard event files.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('eval_results'),
        help='Output directory. Defaults to eval_results.')
    args = parser.parse_args()

    models_dir = args.models_dir
    output_dir = args.output_dir
    event_paths = sorted(models_dir.rglob(EVENT_FILE_PATTERN))

    if not event_paths:
        raise FileNotFoundError(
            'No {} files found under {}'.format(
                EVENT_FILE_PATTERN, models_dir.resolve()))

    event_paths_by_run = defaultdict(list)
    for event_path in event_paths:
        event_paths_by_run[event_path.parent].append(event_path)

    total_plots = 0
    for run_dir, run_event_paths in sorted(event_paths_by_run.items()):
        scalars = merge_scalars(run_event_paths)
        relative_run_dir = run_dir.relative_to(models_dir)
        run_name = safe_filename(str(relative_run_dir)) or safe_filename(models_dir.name)
        run_output_dir = output_dir / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        for tag, values in sorted(scalars.items()):
            output_path = run_output_dir / '{}.png'.format(safe_filename(tag))
            plot_scalar(tag, values, output_path)
            print('Saved {} ({} points)'.format(output_path, len(values)))
            total_plots += 1

        if not scalars:
            print('No scalar summaries found in {}'.format(run_dir))

    print('Generated {} diagrams from {} event files in {} runs.'.format(
        total_plots, len(event_paths), len(event_paths_by_run)))


if __name__ == '__main__':
    main()
