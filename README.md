BankMind - Track B: ML Engineer

Predicts whether a bank customer will subscribe to a term deposit, using the
UCI Bank Marketing dataset (bank-full.csv, 45,211 records, 17 columns).

What's in here


bankmind_track_b.py — full pipeline: EDA → preprocessing → Logistic
Regression baseline → XGBoost main model → evaluation → sample predictions
→ saved model
bank-full.csv — the dataset (CC BY 4.0, included for convenience so the
script runs out of the box)
bankmind_xgb_model.pkl — the trained XGBoost pipeline (preprocessing +
model bundled together via sklearn Pipeline)
sample_predictions.csv — the 5 sample customer predictions printed by
the script
EXPLANATION.md — written answers to the assignment's reflection
questions, based on this run's actual output
requirements.txt — dependencies


How to run

bashpip install -r requirements.txt
python bankmind_track_b.py

That's it — no other setup needed, the dataset is already in this folder.
The script prints EDA output, both models' classification reports, a
side-by-side comparison table, feature importances, 5 sample predictions,
and finally saves the trained model to bankmind_xgb_model.pkl.

A few choices worth flagging


Dropped duration (last call length in seconds). The dataset's own
documentation notes this column leaks the outcome almost directly
(duration = 0 basically guarantees "no"), and it's only known after a
call happens — so it's useless for a model meant to help an RM decide
who to call in the first place. Keeping it would have inflated every
metric for the wrong reasons.
class_weight="balanced" / scale_pos_weight on both models, since
only ~11.7% of customers actually subscribed. Without this, both models
default toward predicting "no" for almost everyone.
Evaluation leads with precision/recall/F1 rather than accuracy, for the
same imbalance reason — see EXPLANATION.md Q1 and Q4 for why.


XGBoost outperforms the baseline on every metric, most notably F1 (0.459 vs
0.373) — better at correctly identifying likely subscribers without
drowning in false positives.# bankmind---Trisha-Kapoor
