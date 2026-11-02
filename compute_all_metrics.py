import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def lcs_length(x, y):
    m = len(x)
    n = len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]

def rouge_l_f1(reference, candidate):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()

    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return 0.0

    lcs = lcs_length(ref_tokens, cand_tokens)
    prec = lcs / float(len(cand_tokens))
    rec = lcs / float(len(ref_tokens))

    if prec + rec == 0:
        return 0.0

    f1 = (2 * prec * rec) / (prec + rec)
    return f1

df = pd.read_csv("results_input_v2.csv")

smooth = SmoothingFunction().method1

bleu1_total = 0.0
bleu2_total = 0.0
bleu3_total = 0.0
bleu4_total = 0.0
rouge_total = 0.0

n = len(df)

for i, row in df.iterrows():
    reference = [row["true_caption"].split()]
    candidate = row["transformer_caption"].split()

    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    rouge_l = rouge_l_f1(row["true_caption"], row["transformer_caption"])

    bleu1_total += bleu1
    bleu2_total += bleu2
    bleu3_total += bleu3
    bleu4_total += bleu4
    rouge_total += rouge_l

    print("Sample {}".format(i + 1))
    print("BLEU-1: {:.4f}".format(bleu1))
    print("BLEU-2: {:.4f}".format(bleu2))
    print("BLEU-3: {:.4f}".format(bleu3))
    print("BLEU-4: {:.4f}".format(bleu4))
    print("ROUGE-L: {:.4f}".format(rouge_l))
    print("-" * 40)

print("\n=== AVERAGE SCORES ===")
print("BLEU-1: {:.4f}".format(bleu1_total / n))
print("BLEU-2: {:.4f}".format(bleu2_total / n))
print("BLEU-3: {:.4f}".format(bleu3_total / n))
print("BLEU-4: {:.4f}".format(bleu4_total / n))
print("ROUGE-L: {:.4f}".format(rouge_total / n))
