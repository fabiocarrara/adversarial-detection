import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import metrics


def main():
    parser = argparse.ArgumentParser(description='Compute adversarial detection metrics from scores')
    parser.add_argument('detection_results', type=str, help='Location to detection scores CSV.')
    parser.add_argument('classification_results', type=str, help='Location to classifications CSV.')
    parser.add_argument('-f', '--feature', type=str, help='Name of the feature used.')
    parser.add_argument('-s', '--scoring', type=str, help='kNN scoring scheme used.')
    parser.add_argument('-p', '--preprocess', type=str, help='Feature preprocessing used.')
    parser.add_argument('-o', '--output', type=str, default='experiments.csv', help='CSV output file.')
    # parser.add_argument('-', '--', type=str, help='.')
    
    args = parser.parse_args()
    
    data = {
        'Feature': args.feature,
        'PreProc': args.preprocess,
        'Scoring': args.scoring
    }
    
    if os.path.exists(args.output): # if exists, load it and check skip
        experiments = pd.read_csv(args.output)
        if (experiments[data.keys()].values == np.array(data.values(), dtype=np.object)).all(1).any():
            print 'SKIPPING Eval: ' + ', '.join(('%s=%s' % (k, v) for k, v in data.iteritems()))
            return
    else:
        experiments = pd.DataFrame()
    
    class_data = pd.read_csv(args.classification_results)
    score_data = pd.read_csv(args.detection_results, names=['AttackName', 'ImageURL', 'PredictedLabel', 'Score'])
    
    class_data = class_data[class_data['DefenseName'] == 'base_inception_model']
    score_data['ImageId'] = score_data['ImageURL'].apply(lambda x: x[:-4])
    
    merged_data = pd.merge(score_data, class_data, on=('AttackName', 'ImageId'), suffixes=('', '_2'))
    merged_data = merged_data[['AttackName', 'ImageId', 'PredictedLabel', 'TrueLabel', 'TargetClass', 'Score']]
    merged_data = merged_data.sort_values(by=['AttackName', 'Score'])
    
    merged_data['IsAuthentic'] = merged_data['PredictedLabel'] == merged_data['TrueLabel']
    y = merged_data['IsAuthentic']
    score = merged_data['Score']
    
    fpr, tpr, thresholds = metrics.roc_curve(y, score)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.absolute(fnr - fpr))
    eer_threshold = thresholds[eer_idx]
    eer_accuracy = metrics.accuracy_score(y, score > eer_threshold)
    
    auth_percentage = y.mean()
    adv_percentage = 1 - auth_percentage
    null_accuracy = max(auth_percentage, adv_percentage)
    print 'Test Set: {:.2f}% Auth, {:.2f}% Adv'.format(100 * auth_percentage, 100 * adv_percentage)
    print 'Null Accuracy: {:.2f}%'.format(100 * null_accuracy)
    print 'EER Accuracy: {:.2f}%'.format(100 * eer_accuracy)
    
    data = dict(data, EER_Accuracy=eer_accuracy, Null_Accuracy=null_accuracy)
    
    # Compute EER Accuracy per attack
    for k, group in merged_data.groupby('IsAuthentic'):
        if k == True: # is authentic
            acc = metrics.accuracy_score(group['IsAuthentic'], group['Score'] > eer_threshold)
            data = dict(data, Auth_Accuracy=acc)
        else: # is adversarial or error
            for attack, attack_group in group.groupby('AttackName'):
                acc = metrics.accuracy_score(attack_group['IsAuthentic'], attack_group['Score'] > eer_threshold)
                data[attack + '_Accuracy'] = acc

#    plt.figure(figsize=(20,7))
#    plt.plot(tpr, thresholds)
#    plt.plot(fpr, thresholds)
#    plt.savefig('tpr-vs-fpr.pdf')
    
    experiments = experiments.append([data], ignore_index=True)
    experiments.to_csv(args.output, index=False)
    

if __name__ == '__main__':
    main()


