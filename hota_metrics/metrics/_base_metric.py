
import numpy as np
from abc import ABC, abstractmethod
from .. import _timing

# Used to ensure namespace of all headers for all metrics is unique.
global_headers = []


class _BaseMetric(ABC):
    @abstractmethod
    def __init__(self):
        self.plottable = False
        self.integer_headers = []
        self.float_headers = []
        self.set_keys = []
        self.integer_set_headers = []
        self.float_set_headers = []
        self.headers = []
        self.summary_headers = []
        self.registered = False

    #####################################################################
    # Abstract functions for subclasses to implement

    @_timing.time
    @abstractmethod
    def eval_sequence(self, data):
        ...

    @abstractmethod
    def combine_sequences(self, all_res):
        ...

    def plot_results(self, all_res, tracker, output_folder, cls):
        """Plot results of metrics, only valid for metrics with self.plottable"""
        if self.plottable:
            raise NotImplementedError
        else:
            pass

    #####################################################################
    # Helper functions which are useful for all metrics:

    @classmethod
    def get_name(cls):
        return cls.__name__

    def register_headers_globally(self):
        global global_headers
        for h in self.headers:
            if h in global_headers:
                raise Exception('metric header %s is defined multiple times by different metrics')
        global_headers += self.headers
        self.registered = True

    @staticmethod
    def _combine_sum(all_res, header):
        """Combine sequence results via sum"""
        return sum([all_res[k][header] for k in all_res.keys()])

    @staticmethod
    def _combine_weighted_av(all_res, header, comb_res, weight_header):
        """Combine sequence results via weighted average"""
        return sum([all_res[k][header] * all_res[k][weight_header] for k in all_res.keys()]) / np.maximum(1.0, comb_res[
            weight_header])

    def print_table(self, table_res, tracker, cls):
        """Prints table of results for all sequences"""
        print('')
        metric_name = self.get_name()
        self._row_print([metric_name + ': ' + tracker + '-' + cls] + self.summary_headers)
        for seq, results in sorted(table_res.items()):
            if seq == 'COMBINED_SEQ':
                continue
            summary_res = self._summary_row(results)
            self._row_print([seq] + summary_res)
        summary_res = self._summary_row(table_res['COMBINED_SEQ'])
        self._row_print(['COMBINED'] + summary_res)

    def _summary_row(self, results_):
        vals = []
        for h in self.summary_headers:
            if h in self.float_set_headers:
                vals.append("{0:1.5g}".format(100 * np.mean(results_[h])))
            elif h in self.float_headers:
                vals.append("{0:1.5g}".format(100 * results_[h]))
            elif h in self.integer_headers:
                vals.append("{0:d}".format(int(results_[h])))
            else:
                raise NotImplementedError("Summary function not implemented for this header type.")
        return vals

    @staticmethod
    def _row_print(*argv):
        """Prints results in an evenly spaced rows, with more space in first row"""
        if len(argv) == 1:
            argv = argv[0]
        to_print = '%-30s' % argv[0]
        for v in argv[1:]:
            to_print += '%-10s' % str(v)
        print(to_print)

    def summary_results(self, table_res):
        """Returns a simple summary of final results for a tracker"""
        # Ensure that header namespace does not have duplicates across metrics.
        assert self.registered, 'self.register_headers_globally() not run for this metric'
        return dict(zip(self.summary_headers, self._summary_row(table_res['COMBINED_SEQ'])))

    def detailed_results(self, table_res):
        """Returns detailed final results for a tracker"""
        # Ensure that header namespace does not have duplicates across metrics.
        assert self.registered, 'self.register_headers_globally() not run for this metric'

        # Get detailed headers
        detailed_headers = self.float_headers + self.integer_headers
        for h in self.float_set_headers + self.integer_set_headers:
            for alpha in [int(100*x) for x in self.set_keys]:
                detailed_headers.append(h + '___' + str(alpha))
            detailed_headers.append(h + '___AUC')

        # Get detailed results
        detailed_results = {}
        for seq, res in table_res.items():
            detailed_row = self._detailed_row(res)
            assert len(detailed_row) == len(detailed_headers), 'Headers and data have different sizes'
            detailed_results[seq] = dict(zip(detailed_headers, detailed_row))
        return detailed_results

    def _detailed_row(self, res):
        detailed_row = []
        for h in self.float_headers + self.integer_headers:
            detailed_row.append(res[h])
        for h in self.float_set_headers + self.integer_set_headers:
            for i, alpha in enumerate([int(100 * x) for x in self.set_keys]):
                detailed_row.append(res[h][i])
            detailed_row.append(np.mean(res[h]))
        return detailed_row
