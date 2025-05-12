# IMPORTS
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

from sklearn.preprocessing import LabelBinarizer

##python3  run_stats.py --dataset /Users/danajulian/Desktop/File\ Registries/binary.csv --k 5 --t wmh_binary --stats_dir /Volumes/One\ Touch/CLAM\ Models/wmh_binary_stats --eval_dir /Volumes/One\ Touch/CLAM\ Models/EVAL_wmh_binary

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type = str,
                    help = "Path to Dataset CSV File")
parser.add_argument('--k', type = int, default = 1,
                    help = "Number of Evaluation Folds to Process")
parser.add_argument('--task', '--t', type = str, default = '',
                    help = "Experiment Name")
parser.add_argument('--stats_dir', type = str,
                    help = "Directory to Save Statistical Output")
parser.add_argument('--eval_dir', type = str,
                    help = "Path to Evaluation Directory for Experiment")

args = parser.parse_args()

# Set Up
print("** SET UP **")

if args.stats_dir is not None:
    save_pre = f'{args.stats_dir}/{args.task}FIG/'
    txt_path = f'{save_pre}{args.task}OUT.txt'
else:
    save_pre = f'{args.task}FIG/'
    txt_path = f'{save_pre}EXP.txt'

if not os.path.exists(save_pre):
    os.mkdir(save_pre)

def category_stats(df:pd.DataFrame, truth_value, outfile):
    tdf = df.where(df["Y"] == truth_value).dropna()

    total = len(tdf)
    correct = 0

    if total == 0:
        outfile.write(f"\nCategory {truth_value} not represented in evalutation data.\n")

    else:
        for idx, row in tdf.iterrows():
            if ( row["Y_hat"] == truth_value ):
                correct += 1

        proportion = correct / total
        percentage = proportion * 100

        outfile.write(f'\nCategory {truth_value} Success Rate: {correct}/{total} ({round(percentage, 2)}%)\n')

with open(txt_path, 'w') as f:
    # Data Prep:
    DF = pd.read_csv(args.dataset)

    n_cases = DF["case_id"].nunique()
    n_slides = DF["slide_id"].nunique()
    n_cat = DF["label"].nunique()

    LABELS = ( DF["label"].unique() ).tolist()
    CODES = ( DF["label_code"].unique() ).tolist()
    LC = dict(zip(CODES, LABELS))

    f.write(f"Experiment {args.task}: \n\n")
    f.write(f"Total Number of Cases: {n_cases}\n")
    f.write(f"Total Number of Slides: {n_slides}\n")

    # Whole Dataset Counts:
    print("** TOTAL COUNTS **")

    cat_counts = DF["label"].value_counts()

    plt.figure()
    plt.bar(cat_counts.keys(), cat_counts.values)
    loc, lab = plt.xticks()
    plt.xticks(loc, cat_counts.keys(), rotation = 30)
    plt.ylabel("Count")
    plt.title("Number of Slides per Category")
    plt.savefig(f'{save_pre}counts.png')
    plt.close()

    # Evaluation Dataset:
    print()
    for k in range(args.k):
        print(f"** EVAL START - FOLD {k} / {args.k - 1} **")

        EDF = pd.read_csv(f"{args.eval_dir}/fold_{k}.csv")
        ELABELS = ( EDF["Y"].unique() ).tolist()
        ELC = { x: LC[x] for x in ELABELS }
        # print(ELC.values())
        f.write(f"\n--------------------------------------\n")
        f.write(f"Evaluation Fold {k}: \n")
        f.write(f"Number of Slides in Evalutation Set {k}: {len(EDF.slide_id)}\n")

        # print("** CATEGORY STATS **")
        for i in range(n_cat):
            category_stats(EDF, i, f)

        # print("** CLASSIFICATION REPORT **")
        f.write(f"\nClassification Report - Evaluation Fold {k}:\n")
        sorted_vals = dict(sorted(ELC.items())).values()
        f.write( classification_report(y_true = EDF["Y"], y_pred = EDF["Y_hat"],
                                       target_names = sorted_vals,
                                       zero_division = 0.0) )
        f.write("\n")

        # Evaluation Counts:
        # print("** COUNTS **")
        e_counts = EDF["Y"].value_counts()

        plt.figure()
        plt.bar(e_counts.keys(), e_counts.values)
        plt.xticks(ELABELS, ELC.values(), rotation = 30)
        plt.ylabel("Count")
        plt.title(f"Eval Fold {k}\nNumber of Slides per Category")
        plt.savefig(f"{save_pre}ecounts{k}.png")
        plt.close()

        # Confusion Matrix:
        # print("** CONFUSION **")
        cm = confusion_matrix(y_true = EDF["Y"], y_pred = EDF["Y_hat"])
        cm_disp = ConfusionMatrixDisplay(cm, display_labels = sorted_vals).plot()
        cm_disp.ax_.set_title(f"Eval Fold {k}")
        plt.savefig(f"{save_pre}CM{k}.png")
        plt.close()

        # ROC Curve:
        # print("** ROC **")
        LB = LabelBinarizer().fit(EDF["Y"])
        y_OH = LB.transform(EDF["Y"])

        y_scores = np.array( EDF[EDF.columns[4:]] )

        fprm, tprm, _ =  roc_curve(y_OH.ravel(), y_scores.ravel())
        aucm = auc(fprm, tprm)

        fig, ax = plt.subplots()

        plt.plot(
            fprm,
            tprm,
            label = f"Micro-Avg (AUC = {aucm:.2f})",
            color = "gray",
            linestyle = ":"
        )

        for cid in range(len(ELABELS)):
            RocCurveDisplay.from_predictions(
                # y_OH[:, cid],
                # y_scores[:, cid],
                #name = ELC[cid]
                y_OH,
                y_scores,
                #changed by Dana 5/16/24
                name = 'Fold',
                #name = ELC[0],
                ax = ax
            )

        plt.title(f"Eval Fold {k} ROC")
        plt.legend(bbox_to_anchor = (1, 0.5))
        plt.savefig(f"{save_pre}ROC{k}.png", bbox_inches = "tight")
        plt.close()

        # Precision Recall:
        # print("** PR **")
        P, R, P_avg = dict(), dict(), dict()

        # for i in range(len(ELABELS)):
        for i in range(0,1):
            P[i], R[i], _ = precision_recall_curve(y_OH[:, i], y_scores[:, i])
            P_avg[i] = average_precision_score(y_OH[:, i], y_scores[:, i])

        P['micro'], R['micro'], _ = precision_recall_curve(y_OH.ravel(), y_scores.ravel())
        P_avg['micro'] = average_precision_score(y_OH, y_scores, average = 'micro')

        fig, ax = plt.subplots()

        pr_disp = PrecisionRecallDisplay(
            recall = R["micro"],
            precision = P["micro"],
            average_precision = P_avg['micro']
        )
        pr_disp.plot(ax = ax, name = 'micro-avg', color = 'gray')

        # for i in range(len(ELABELS)):
        for i in range(0,1):
            pr_disp = PrecisionRecallDisplay(
                recall = R[i],
                precision = P[i],
                average_precision = P_avg[i]
            )
            pr_disp.plot(ax = ax, name = "Fold")

        plt.title(f"Eval Fold {k}:\nPrecision v Recall")
        plt.legend(bbox_to_anchor = (1, 0.5))
        plt.savefig(f"{save_pre}PR{k}.png", bbox_inches = "tight")
        plt.close()

        print()

print("Done.")
