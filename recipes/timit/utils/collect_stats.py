#!/usr/bin/env python
import os
import glob
import argparse


def read_nmi(file):
    experiment = file.split('/')[-3]
    with open(file) as fp:
        lines = fp.readlines()
        for line in lines:
            line_split = line.split()
            if line_split[0] == 'I(X;Y)/(H(X)):':
                return (experiment, float(line_split[1]))


def read_abx_scores(file):
    output_measures = {'within_talkers', 'across_talkers'}
    experiment = file.split('/')[-3]
    try:
        with open(file) as fp:
            lines = fp.readlines()
            result = [float(line_split[1]) for line_split in [line.split() for line in lines]
                     if len(line_split) > 1 and len(line_split[0]) > 13 and line_split[0][-15:-1] in output_measures]
            return (experiment, result)
    except IOError:
            return (experiment, [])


def read_eval2_measure(file):
    output_measures = {'precision', 'recall', 'fscore', 'NED', 'coverage'}
    experiment = '_'.join(file.split('/')[-4:-2])
    try:
        with open(file) as fp:
            lines = fp.readlines()
            result = [float(line_split[1]) for line_split in [line.split() for line in lines]
                     if len(line_split) > 1 and line_split[0] in output_measures]
            return (experiment, result)
    except IOError:
            return (experiment, [])


def main():
    # paramter parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', help='Experiment directory')
    parser.add_argument('--nmi', action='store_true', help='Collect NMI stats')
    parser.add_argument('--eval1', action='store_true', help='Colltct eval1 results')
    parser.add_argument('--eval2', action='store_true', help='Collect eval2 results')
    args = parser.parse_args()

    experiment_dir = args.experiment_dir

    baseline_header = ['Experiment']

    if args.nmi:
        # nmi stats
        unigram_labels_nmi_file = os.path.join(experiment_dir, 'unigram_labels_nmi/scores')
        unigram_lattices_nmi_file = os.path.join(experiment_dir, 'unigram_lattices_nmi/scores')
        bigram_labels_nmi_file = os.path.join(experiment_dir, 'bigram_labels_nmi/scores')
        bigram_lattices_nmi_file = os.path.join(experiment_dir, 'bigram_lattices_nmi/scores')
        baseline_header += ['NMI (labels)', 'NMI (lattices)']

        baseline_nmi = {'unigram': [read_nmi(unigram_labels_nmi_file)[1], read_nmi(unigram_lattices_nmi_file)[1]],
                        'bigram': [read_nmi(bigram_labels_nmi_file)[1], read_nmi(bigram_lattices_nmi_file)[1]]}

    if args.eval1:
        # eval 1 evaluation
        unigram_states_abx_file = os.path.join(experiment_dir, 'unigram_eval1/unigram_gmm_posts_txt/results.txt')
        unigram_units_abx_file = os.path.join(experiment_dir, 'unigram_eval1/unigram_unit_gmm_posts_txt/results.txt')
        bigram_states_abx_file = os.path.join(experiment_dir, 'bigram_eval1/bigram_gmm_posts_txt/results.txt')
        bigram_units_abx_file = os.path.join(experiment_dir, 'bigram_eval1/bigram_unit_gmm_posts_txt/results.txt')
        baseline_header += ['State ABX (within)', 'State ABX (across)',
                            'Unit ABX (within)', 'Unit ABX (across)']

        baseline_abx = {'unigram': read_abx_scores(unigram_states_abx_file)[1] + read_abx_scores(unigram_units_abx_file)[1],
                        'bigram': read_abx_scores(bigram_states_abx_file)[1] + read_abx_scores(bigram_units_abx_file)[1]}

    # eval 2 evaluation
    if args.eval2:
        eval2_base = 'bigram_ws_eval2/bigram_ws/*/*/TimedSentences_Iter_150'
        eval2_directories = glob.glob(os.path.join(experiment_dir, eval2_base))

        eval2_measures = ['boundary', 'group', 'matching', 'nlp', 'token_type']
        result_header = ['Experiment',
                        'Boundary precision (total)', 'Boundary recall (total)', 'Boundary fscore (total)',
                        'Boundary precision (within)', 'Boundary recall (within)', 'Boundary fscore (within)',
                        'Grouping precision (total)', 'Grouping recall (total)', 'Grouping fscore (total)',
                        'Grouping precision (within)', 'Grouping recall (within)', 'Grouping fscore (within)',
                        'Matching precision (total)', 'Matching recall (total)', 'Matching fscore (total)',
                        'Matching precision (within)', 'Matching recall (within)', 'Matching fscore (within)',
                        'NED (total)', 'Coverage (total)',
                        'NED (within)', 'Coverage (within)',
                        'Token precision (total)', 'Token recall (total)', 'Token fscore (total)',
                        'Type precision (total)', 'Type recall (total)', 'Type fscore (total)',
                        'Token precision (within)', 'Token recall (within)', 'Token fscore (within)',
                        'Type precision (within)', 'Type recall (within)', 'Type fscore (within)']

        eval2_results =  {measure: {experiment: result for experiment, result in 
                                    [read_eval2_measure(os.path.join(directory, measure)) for directory in eval2_directories]}
                          for measure in eval2_measures}

    # summarize results
    if args.eval2:
        results = dict()

        for measure in eval2_measures:
            for experiment, result in eval2_results[measure].items():
                if len(result) == 0:
                    if measure == 'nlp':
                        result = [1, 0]*2
                    else:
                        result = [0]*6
                try:
                    results[experiment] += result
                except KeyError:
                    results[experiment] = result
                    
    if args.nmi and args.eval1:
        baseline_results = {experiment: baseline_nmi[experiment] + baseline_abx[experiment] for experiment in baseline_nmi}
    elif args.nmi:
        baseline_results = baseline_nmi
    elif args.eval1:
        baseline_results = baseline_abx
    else:
        baseline_results = dict()

    # write out results
    with open(os.path.join(experiment_dir, 'results.csv'), 'w') as fid:
        if args.eval2:
            string_format = '{}' + ';{}'*(len(result_header) - 1) + '\n'
            fid.write(string_format.format(*result_header))
            for experiment, result in results.items():
                fid.write(string_format.format(experiment, *result))

            fid.write('\n')

        if args.nmi or args.eval1:
            string_format = '{}' + ';{}'*(len(baseline_header) - 1) + '\n'
            fid.write(string_format.format(*baseline_header))
            for experiment, result in baseline_results.items():
                fid.write(string_format.format(experiment, *result))


if __name__ == '__main__':
    main()
