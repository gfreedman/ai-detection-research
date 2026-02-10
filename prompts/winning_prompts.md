# Winning Prompts — What Actually Works

These are the exact prompts that reduce ZeroGPT detection from ~100% to near-zero.
Results from 700 experiment runs against ZeroGPT using Gemini 2.0 Flash.

---

## Best Single Variant: P3b "Varied Sentence Length"

**Mean AI detection: 9.5% | Pass rate: 84%**

This single instruction drops detection by ~90 percentage points.

**System prompt:** _(none)_

**User prompt:**
```
Write a 500-word essay about [TOPIC].
Vary sentence length dramatically — mix 4-word fragments with 35-word run-ons.
```

**Generation params:** temperature=1.5, top_p=1.0

---

## Best Composite: "All 5 Winners"

**Mean AI detection: 1.2% | Pass rate: 100%**

Combines the top performer from each of the 5 prompt dimensions.

### System Prompt

```
You are a B+ student who writes well but not perfectly. You sometimes lose your
train of thought and your conclusions tend to trail off.
```

### User Prompt

```
Write a 500-word essay about [TOPIC]. Use 2-3 very short paragraphs and one long
rambling one. Vary sentence length dramatically — mix 4-word fragments with
35-word run-ons. Mention a real but obscure source you 'read in class' as if
recalling it from memory. You're writing this the night before it's due. You're
tired and rushing a bit.
```

### Generation Parameters

```
temperature: 1.5
top_p: 1.0
```

### What Each Component Contributes

| Component | Dimension | Contribution | Solo Performance |
|-----------|-----------|-------------|------------------|
| **P1c** | Persona | System prompt (B+ student voice) | 11.9% mean, 76% pass |
| **P2d** | Structure | "Use 2-3 very short paragraphs and one long rambling one." | 10.5% mean, 80% pass |
| **P3b** | Texture | "Vary sentence length dramatically..." | 9.5% mean, 84% pass |
| **P4c** | Content | "Mention a real but obscure source..." | 29.0% mean, ~28% pass |
| **P5c** | Meta | "You're writing this the night before..." | 31.0% mean, ~24% pass |

---

## Runner-Up Composites

### "Top 3" (P1c + P2d + P3b)

**Mean AI detection: 2.8% | Pass rate: 100%**

Uses only the three dimensions that individually beat the 15% threshold.

**System prompt:** Same as above (P1c).

**User prompt:**
```
Write a 500-word essay about [TOPIC]. Use 2-3 very short paragraphs and one long
rambling one. Vary sentence length dramatically — mix 4-word fragments with
35-word run-ons.
```

### "Top 3 + Meta" (P1c + P2d + P3b + P5c)

**Mean AI detection: 2.0% | Pass rate: 100%**

**System prompt:** Same as above (P1c).

**User prompt:**
```
Write a 500-word essay about [TOPIC]. Use 2-3 very short paragraphs and one long
rambling one. Vary sentence length dramatically — mix 4-word fragments with
35-word run-ons. You're writing this the night before it's due. You're tired and
rushing a bit.
```

---

## Why These Work (Hypothesis)

ZeroGPT appears to rely heavily on two signals:

1. **Burstiness** — variance in sentence complexity. AI text tends to produce
   uniformly medium-length, well-structured sentences. The "varied sentence
   length" instruction (P3b) directly disrupts this signal.

2. **Structural regularity** — predictable paragraph patterns. The "irregular
   paragraphs" instruction (P2d) breaks the 5-paragraph essay template that
   AI defaults to.

3. **Persona framing** shifts the model's entire output distribution. Telling
   it to "write like a B+ student" produces less polished, more naturalistic
   text that doesn't trigger the "too perfect" heuristic.

Temperature (1.5) adds lexical randomness but is much less effective alone
(only drops to ~82% detection). The prompt engineering is what matters.
