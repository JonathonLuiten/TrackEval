import numpy as np
from copy import deepcopy as copy
from matplotlib import pyplot as plt
from lapjv import lapjv as hungarian # 8x faster than scipy
import csv
import os

def combine_sequences(res_list, metrics):
    combined_res = {}

    if 'HOTA' in metrics:
        combined_res['HOTA_TP'] = sum([res_list[k]['HOTA_TP'] for k in res_list.keys()])
        combined_res['HOTA_FP'] = sum([res_list[k]['HOTA_FP'] for k in res_list.keys()])
        combined_res['HOTA_FN'] = sum([res_list[k]['HOTA_FN'] for k in res_list.keys()])

        combined_res['DetRe'] = combined_res['HOTA_TP'] / (combined_res['HOTA_TP'] + combined_res['HOTA_FN'])
        combined_res['DetPr'] = combined_res['HOTA_TP'] / (combined_res['HOTA_TP'] + combined_res['HOTA_FP'])
        combined_res['DetA'] = combined_res['HOTA_TP'] / (combined_res['HOTA_TP'] + combined_res['HOTA_FN'] + combined_res['HOTA_FP'])

        combined_res['AssRe'] = sum([res_list[k]['AssRe'] * res_list[k]['HOTA_TP'] for k in res_list.keys()]) / combined_res['HOTA_TP']
        combined_res['AssPr'] = sum([res_list[k]['AssPr'] * res_list[k]['HOTA_TP'] for k in res_list.keys()]) / combined_res['HOTA_TP']
        combined_res['AssA'] = sum([res_list[k]['AssA'] * res_list[k]['HOTA_TP'] for k in res_list.keys()]) / combined_res['HOTA_TP']
        combined_res['LocA'] = sum([res_list[k]['LocA'] * res_list[k]['HOTA_TP'] for k in res_list.keys()]) / combined_res['HOTA_TP']

        combined_res['HOTA'] = np.sqrt(combined_res['DetA'] * combined_res['AssA'])

    if 'CLEAR' in metrics:
        combined_res['CLR_TP'] = sum([res_list[k]['CLR_TP'] for k in res_list.keys()])
        combined_res['CLR_FP'] = sum([res_list[k]['CLR_FP'] for k in res_list.keys()])
        combined_res['CLR_FN'] = sum([res_list[k]['CLR_FN'] for k in res_list.keys()])
        combined_res['IDSW'] = sum([res_list[k]['IDSW'] for k in res_list.keys()])
        combined_res['MT'] = sum([res_list[k]['MT'] for k in res_list.keys()])
        combined_res['PT'] = sum([res_list[k]['PT'] for k in res_list.keys()])
        combined_res['ML'] = sum([res_list[k]['ML'] for k in res_list.keys()])
        combined_res['Frag'] = sum([res_list[k]['Frag'] for k in res_list.keys()])

        combined_res['MODA'] = (combined_res['CLR_TP'] - combined_res['CLR_FP']) / (combined_res['CLR_TP'] + combined_res['CLR_FN'])
        combined_res['MOTA'] = (combined_res['CLR_TP'] - combined_res['CLR_FP'] - combined_res['IDSW']) / (combined_res['CLR_TP'] + combined_res['CLR_FN'])
        combined_res['Recall'] = combined_res['CLR_TP'] / (combined_res['CLR_TP'] + combined_res['CLR_FN'])
        combined_res['Precision'] = combined_res['CLR_TP'] / (combined_res['CLR_TP'] + combined_res['CLR_FP'])

        combined_res['MOTP'] = sum([res_list[k]['MOTP'] * res_list[k]['CLR_TP'] for k in res_list.keys()]) / combined_res['CLR_TP']

    if 'ID' in metrics:
        combined_res['IDTP'] = sum([res_list[k]['IDTP'] for k in res_list.keys()])
        combined_res['IDFP'] = sum([res_list[k]['IDFP'] for k in res_list.keys()])
        combined_res['IDFN'] = sum([res_list[k]['IDFN'] for k in res_list.keys()])

        combined_res['IDR'] = combined_res['IDTP'] / (combined_res['IDTP'] + combined_res['IDFN'])
        combined_res['IDP'] = combined_res['IDTP'] / (combined_res['IDTP'] + combined_res['IDFP'])
        combined_res['IDF1'] = combined_res['IDTP'] / (combined_res['IDTP'] + 0.5*combined_res['IDFP'] + 0.5*combined_res['IDFN'])

    return combined_res

def squarify(M,val):
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)

def linear_assignment_problem(score_mat):
    match_cols, _, _ = hungarian(-score_mat)
    match_rows = np.arange(len(match_cols))
    actually_matched_mask = score_mat[match_rows, match_cols] > 0
    match_rows = match_rows[actually_matched_mask]
    match_cols = match_cols[actually_matched_mask]
    return match_rows,match_cols

def myprint(*argv):
    if len(argv)==1:
        argv = argv[0]
    args = copy(argv)
    to_print = '%-16s' % argv[0]
    for v in args[1:]:
        v = str(v)
        to_print += '%-10s'%v
    print(to_print)

def print_results(all_res, tracker, metrics, alpha_behaviour = 'default', print_only_combined = False):
    print('')
    print('#' * 10 + ' ' + tracker + ' ' + '#' * 10)

    all_res = copy(all_res)
    all_res = convert_floats_for_print(all_res)
    print(metrics)
    for metric in metrics:
        if metric in 'HOTA':
            TO_PRINT = {'HOTA_AUC':['HOTA','DetA','AssA','DetRe','DetPr','AssRe','AssPr','LocA'],
                        'HOTA_50': ['HOTA','DetA','AssA','DetRe','DetPr','AssRe','AssPr','LocA','HOTA_TP','HOTA_FP','HOTA_FN']}
            if alpha_behaviour == 'fifty_only': del TO_PRINT['HOTA_AUC']
        elif metric in 'CLEAR':
            TO_PRINT = {'CLEAR_AUC':['MOTA','MOTP','MODA','Recall','Precision'],
                        'CLEAR_50': ['MOTA','MOTP','MODA','Recall','Precision','CLR_TP','CLR_FP','CLR_FN','IDSW','MT','PT','ML','Frag']}
            if alpha_behaviour == 'fifty_only': del TO_PRINT['CLEAR_AUC']
        elif metric in 'ID':
            TO_PRINT = {'ID_AUC':['IDF1','IDP','IDR'],
                        'ID_50': ['IDF1','IDP','IDR','IDTP','IDFP','IDFN']}
            if alpha_behaviour == 'fifty_only': del TO_PRINT['ID_AUC']
        else: TO_PRINT = dict()

        for table_name,metric_headers in TO_PRINT.items():
            print('')
            print(table_name)
            myprint([''] + metric_headers)

            if not print_only_combined:
                for seq,results in sorted(all_res.items()):
                    if seq=='COMBINED': continue
                    vals = [seq]

                    for x in metric_headers:
                        if 'AUC' in table_name:
                            vals.append(np.round(np.mean(results[x]),3))
                        elif alpha_behaviour == 'fifty_only':
                            vals.append(results[x][0])
                        elif alpha_behaviour == 'default':
                            vals.append(results[x][9])
                    myprint(vals)

            vals = ['COMBINED']
            for x in metric_headers:
                if 'AUC' in table_name:
                    vals.append(np.round(np.mean(all_res['COMBINED'][x]),3))
                elif alpha_behaviour == 'fifty_only':
                    vals.append(all_res['COMBINED'][x][0])
                elif alpha_behaviour == 'default':
                    vals.append(all_res['COMBINED'][x][9])
            myprint(vals)

def convert_floats_for_print(all_res):
    floats = ['HOTA','DetA','AssA','DetRe','DetPr','AssRe','AssPr','LocA','MOTA','MOTP','MODA','Recall','Precision','IDF1','IDP','IDR']
    for k,v in all_res.items():
        for k2,v2 in v.items():
            all_res[k][k2] = []
            if k2 in floats:
                for x in v2:
                    if x > 0.9995:
                        x = round(x * 100, 2)
                    elif x < 0.1:
                        x = round(x * 100, 4)
                    else:
                        x = round(x * 100, 3)
                    all_res[k][k2].append(x)
            else:
                for x in v2:
                    all_res[k][k2].append(int(x))
    return all_res

def output_csv(all_res,tracker,csv_out_file,output_folder,alpha_behaviour,csv_out_mode,tracker_iteration):
    metrics = all_res['COMBINED'].keys()
    out_file = os.path.join(output_folder, csv_out_file)
    if csv_out_mode == 'new' and tracker_iteration==0:
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            headers = ['tracker', 'seq']
            for m in metrics:
                if alpha_behaviour == 'default':
                    for x in np.arange(5, 100, 5):
                        headers.append(m + '_' + str(x))
                    headers.append(m + '_AUC')
                elif alpha_behaviour == 'fifty_only':
                    headers.append(m + '_50')
            writer.writerow(headers)

    with open(out_file, 'a') as f:
        writer = csv.writer(f)
        for seq,res in all_res.items():
            if seq == 'COMBINED': continue
            row = []
            row.append(tracker)
            row.append(seq)
            for m in metrics:
                if alpha_behaviour == 'default':
                    for x in np.arange(19):
                        row.append(res[m][x])
                    row.append(np.mean(res[m]))
                elif alpha_behaviour == 'fifty_only':
                    row.append(res[m][0])
            writer.writerow(row)
        row = []
        row.append(tracker)
        row.append('COMBINED')
        for m in metrics:
            if alpha_behaviour == 'default':
                for x in np.arange(19):
                    row.append(res[m][x])
                row.append(np.mean(res[m]))
            elif alpha_behaviour == 'fifty_only':
                row.append(res[m][0])
        writer.writerow(row)


def plot_results(all_res,alpha_thresholds,tracker,metrics,output_folder):
    res = all_res['COMBINED']
    plt.plot((0,0),'w')
    if 'HOTA' in metrics:
        plt.plot(alpha_thresholds, res['HOTA'],'r')
        plt.plot(alpha_thresholds, res['DetA'],'b')
        plt.plot(alpha_thresholds, res['DetRe'],'b--')
        plt.plot(alpha_thresholds, res['DetPr'],'b:')
        plt.plot(alpha_thresholds, res['AssA'], 'g')
        plt.plot(alpha_thresholds, res['AssRe'],'g--')
        plt.plot(alpha_thresholds, res['AssPr'],'g:')
        plt.plot(alpha_thresholds, res['LocA'], 'm')

    if 'CLEAR' in metrics:
        plt.plot(alpha_thresholds, res['MODA'], 'k--')
        plt.plot(alpha_thresholds, res['MOTA'], 'k')
    if 'ID' in metrics:
        plt.plot(alpha_thresholds, res['IDF1'], 'c')
        plt.plot(alpha_thresholds, res['IDR'], 'y--')
        plt.plot(alpha_thresholds, res['IDP'], 'y:')

    plt.xlabel('alpha')
    plt.ylabel('score')
    plt.title(tracker)
    plt.axis([0, 1, 0, 1])
    legend = ['              AUC (@0.5)']
    if 'HOTA' in metrics:
        legend+=[
                'HOTA  = '+str(round(np.mean(res['HOTA']),2))+' ('+str(round(res['HOTA'][9],2))+')',
                'DetA  = '+str(round(np.mean(res['DetA']),2))+' ('+str(round(res['DetA'][9],2))+')',
                'DetRe = '+str(round(np.mean(res['DetRe']),2))+' ('+str(round(res['DetRe'][9],2))+')',
                'DetPr  = '+str(round(np.mean(res['DetPr']),2))+' ('+str(round(res['DetPr'][9],2))+')',
                'AssA  = '+str(round(np.mean(res['AssA']),2))+' ('+str(round(res['AssA'][9],2))+')',
                'AssRe = '+str(round(np.mean(res['AssRe']),2))+' ('+str(round(res['AssRe'][9],2))+')',
                'AssPr  = '+str(round(np.mean(res['AssPr']),2))+' ('+str(round(res['AssPr'][9],2))+')',
                'LocA  = '+str(round(np.mean(res['LocA']), 2))+' ('+str(round(res['LocA'][9], 2))+')']
    if 'CLEAR' in metrics:
        legend += [
                'MODA = ' + str(round(np.mean(res['MODA']), 2))+' ('+str(round(res['MODA'][9],2))+')',
                'MOTA  = ' + str(round(np.mean(res['MOTA']), 2))+' ('+str(round(res['MOTA'][9],2))+')']
    if 'ID' in metrics:
        legend += [
                'IDF1 = ' + str(round(np.mean(res['IDF1']), 2))+' ('+str(round(res['IDF1'][9],2))+')',
                'IDR  = ' + str(round(np.mean(res['IDR']), 2))+' ('+str(round(res['IDR'][9],2))+')',
                'IDP  = ' + str(round(np.mean(res['IDP']), 2))+' (' + str(round(res['IDP'][9], 2))+')']
    plt.legend(legend,loc='lower left')
    out_file = os.path.join(output_folder, tracker + '.pdf')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file)
    plt.clf()


class dotdict(dict):
    """dot.notation access to dictionary attributes - for config file simplicity"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
