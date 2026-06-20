1. What percentage of customers have y = yes? What does this imbalance mean for evaluation?

11.70% of customers subscribed (5,289 out of 45,211); 88.30% did not. This is a strongly
imbalanced dataset, which means accuracy is a misleading metric on its own — a model
that just predicts "no" for every single customer would already score 88.3% accuracy
while being completely useless for the actual business goal (finding the customers worth
calling). That's exactly why I'm reporting precision, recall, and F1 instead of leading
with accuracy, and why I used class_weight="balanced" for Logistic Regression and
scale_pos_weight for XGBoost — without that adjustment, both models would lean heavily
toward predicting "no" since it's the easy, frequent answer.

2. Which job category had the highest subscription rate? Does this make sense intuitively?

Student had the highest subscription rate at 28.68%, followed by retired at 22.79%.
Both sit well above the dataset's overall 11.7% rate, while blue-collar workers had the
lowest at 7.27%.

This does make intuitive sense once you think about who's actually free and financially
flexible enough to engage with a phone-based marketing call: students and retirees
typically have more flexible schedules to actually take the call, fewer existing financial
commitments (mortgages, active loans) competing for their money, and in the case of
retirees specifically, often have lump sums (pensions, savings) they're actively looking to
park safely — a term deposit is a natural fit. Blue-collar and services workers, by
contrast, are more likely to be juggling existing loan repayments and have less
discretionary cash to lock away.

3. Which feature had the highest importance in your tree-based model? Why?

By a wide margin, it was poutcome_success (importance ≈ 0.30, more than double the
next feature). This is the one-hot encoded flag for "this customer said yes to a previous
marketing campaign." It dominating makes sense — past behavior is usually the strongest
predictor of future behavior. A customer who has already trusted the bank enough to buy a
product once is a fundamentally warmer lead than someone being contacted cold, regardless
of their age, job, or balance. The next-most-important features were contact_unknown
(whether we even have a valid contact channel) and several month_* flags, suggesting
campaign timing/season also carries real signal — likely correlated with which kinds of
campaigns ran when.

4. Why is F1 a better metric than accuracy for this dataset?

Because the classes are so imbalanced (88/12), accuracy can stay high while the model
completely fails at its actual job — finding the rare "yes" customers. F1 is the harmonic
mean of precision and recall, so it only stays high if the model is both correctly
flagging "yes" customers when it says yes (precision) and not missing most of the actual
"yes" customers (recall). My XGBoost model has 82.3% accuracy but only 46% F1 on the "yes"
class — that gap is the tell. If I'd only reported accuracy, the model would look great
when it's actually still missing more than a third of real subscribers and over half its
"yes" predictions are false alarms. For a business use case where the cost of an RM calling
the wrong person is low but missing a real lead is costly, F1 (and recall in particular) is
a far more honest signal of usefulness than accuracy.

5. Pick one of your 5 sample predictions — do you agree with the model's call?

Customer at index 12669: age 44, technician, married, tertiary education, balance €1,146,
no housing loan, no personal loan. Model predicted "yes" with 55.9% probability — actual
outcome was "no."

Looking at this one, I think the model's reasoning is defensible even though it was wrong.
This customer has no existing housing or personal loan (meaning more disposable income and
no competing debt obligations), a comfortably positive balance, and a stable
tertiary-educated, married profile — all of which line up with the kind of customer who
tends to convert. But 55.9% is barely above the decision threshold of 50%, so the model
itself wasn't confident; it's essentially saying "this is a coin flip, slightly favoring
yes." Looking at it, I don't think the model is wrong to call this a borderline lead — it
correctly identified someone with favorable financial characteristics. It just couldn't
capture whatever made this specific person say no on this specific call (timing,
conversation quality, an unmodeled personal circumstance). This is honestly the case I'd
expect a Relationship Manager to actually call — not a high-confidence miss, but a
reasonable "worth trying" lead, which is exactly the kind of prioritization signal the
business wants out of this model even when any individual call doesn't convert.
