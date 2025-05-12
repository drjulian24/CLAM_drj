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

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,
                    help="Path to Dataset CSV File")
parser.add_argument('--k', type=int, default=1,
                    help="Number of Evaluation Folds to Process")
parser.add_argument('--task', '--t', type=str, default='',
                    help="Experiment Name")
parser.add_argument('--stats_dir', type=str,
                    help="Directory to Save Statistical Output")
parser.add_argument('--eval_dir', type=str,
                    help="Path to Evaluation Directory for Experiment")

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

def category_stats(df: pd.DataFrame, truth_value, outfile):
    tdf = df.where(df["Y"] == truth_value).dropna()

    total = len(tdf)
    correct = 0

    if total == 0:
        outfile.write(f"\nCategory {truth_value} not represented in evaluation data.\n")
    else:
        for idx, row in tdf.iterrows():
            if row["Y_hat"] == truth_value:
                correct += 1

        proportion = correct / total
        percentage = proportion * 100

        outfile.write(f'\nCategory {truth_value} Success Rate: {correct}/{total} ({round(percentage, 2)}%)\n')

# Lists to hold overall predictions and true labels
overall_Y_hat = []
overall_Y = []

with open(txt_path, 'w') as f:
    # Data Prep:
    DF = pd.read_csv(args.dataset)

    n_cases = DF["case_id"].nunique()
    n_slides = DF["slide_id"].nunique()
    n_cat = DF["label"].nunique()

    LABELS = DF["label"].unique().tolist()
    CODES = DF["label_code"].unique().tolist()
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
    plt.xticks(loc, cat_counts.keys(), rotation=30)
    plt.ylabel("Count")
    plt.title("Number of Slides per Category")
    plt.savefig(f'{save_pre}counts.png')
    plt.close()

    # Evaluation Dataset:
    print()
    for k in range(args.k):
        print(f"** EVAL START - FOLD {k} / {args.k - 1} **")

        EDF = pd.read_csv(f"{args.eval_dir}/fold_{k}.csv")
        ELABELS = EDF["Y"].unique().tolist()
        ELC = {x: LC[x] for x in ELABELS}

        f.write(f"\n--------------------------------------\n")
        f.write(f"Evaluation Fold {k}: \n")
        f.write(f"Number of Slides in Evaluation Set {k}: {len(EDF.slide_id)}\n")

        for i in range(n_cat):
            category_stats(EDF, i, f)

        f.write(f"\nClassification Report - Evaluation Fold {k}:\n")
        sorted_vals = dict(sorted(ELC.items())).values()
        f.write(classification_report(y_true=EDF["Y"], y_pred=EDF["Y_hat"],
                                      target_names=sorted_vals,
                                      zero_division=0.0))
        f.write("\n")

        e_counts = EDF["Y"].value_counts()

        plt.figure()
        plt.bar(e_counts.keys(), e_counts.values)
        plt.xticks(ELABELS, ELC.values(), rotation=30)
        plt.ylabel("Count")
        plt.title(f"Eval Fold {k}\nNumber of Slides per Category")
        plt.savefig(f"{save_pre}ecounts{k}.png")
        plt.close()

        cm = confusion_matrix(y_true=EDF["Y"], y_pred=EDF["Y_hat"])
        cm_disp = ConfusionMatrixDisplay(cm, display_labels=sorted_vals).plot()
        cm_disp.ax_.set_title(f"Eval Fold {k}")
        plt.savefig(f"{save_pre}CM{k}.png")
        plt.close()

        LB = LabelBinarizer().fit(EDF["Y"])
        y_OH = LB.transform(EDF["Y"])

        y_scores = np.array(EDF[EDF.columns[4:]])

        overall_Y_hat.extend(y_scores)
        overall_Y.extend(y_OH)

        print()

    from sklearn.metrics import classification_report

    # Compute overall metrics
    overall_Y_hat = np.array(overall_Y_hat)
    overall_Y = np.array(overall_Y)

    # Compute ROC curve for overall data
    fpr, tpr, _ = roc_curve(overall_Y.ravel(), overall_Y_hat.ravel())

    # Sort fpr and tpr along with thresholds
    sort_inds = np.argsort(fpr)
    fpr_sorted = fpr[sort_inds]
    tpr_sorted = tpr[sort_inds]

    # Compute AUC using sorted arrays
    overall_auc = auc(fpr_sorted, tpr_sorted)

    # Print overall metrics to the text file
    f.write(f"\n\nOverall Classification Report:\n{classification_report(overall_Y.ravel(), overall_Y_hat.ravel())}\n")
    f.write(f"\nOverall AUC: {overall_auc:.2f}\n")

    # Plot overall ROC curve
    plt.figure()
    plt.plot(fpr_sorted, tpr_sorted, color='darkorange', lw=2, label=f"Overall ROC curve (area = {overall_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Overall ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{save_pre}overall_ROC.png", bbox_inches="tight")
    plt.close()

print("Done.")
