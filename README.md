# MBAX 6418: Assignment 1: Sentiment & Emotion Classification of Amazon Reviews

## 1. Project goal

The goal of this project was to build an LLM-based classifier for Amazon reviews and evaluate how well it could identify both sentiment and emotion.

The model only received the review **title and text**. It did not receive the star rating. The rating was only used afterward to check whether the model's prediction matched the rating-based class.

The project started with binary sentiment classification:

- **POSITIVE**
- **NEGATIVE**

Later, the classification was expanded to three classes:

- **POSITIVE**
- **NEUTRAL**
- **NEGATIVE**

I also compared two different ways of identifying the primary emotion in a review:

1. An LLM-based prediction
2. A word-list approach using the NRC Emotion Lexicon

The eight emotion categories used were anger, anticipation, disgust, fear, joy, sadness, surprise, and trust.

All LLM calls used the professor-provided OpenAI-compatible endpoint. The API information was stored in a local `.env` file so the key would not be included in the project code or committed to GitHub.

---

## 2. Dataset

This project used the **Amazon Reviews '23 Gift Cards dataset** from the McAuley Lab.

The Gift Cards file contains **152,410 reviews**. The main fields used in this project were:

- `title`
- `text`
- `rating`

Other available fields included `asin`, `parent_asin`, `user_id`, `timestamp`, `helpful_vote`, `verified_purchase`, and `images`.

The original dataset link provided in the assignment returned a 404 error, so I used the official McAuley Lab Hugging Face mirror instead. The file was saved locally as `data/Gift_Cards.jsonl.gz`.

The full dataset was very imbalanced:

| Class | Reviews | Share |
| :--- | ---: | ---: |
| POSITIVE (4–5★) | 134,940 | 88.5% |
| NEUTRAL (3★) | 3,271 | 2.1% |
| NEGATIVE (1–2★) | 14,199 | 9.3% |

This imbalance became important later because the first classification run looked much stronger than the balanced run.

---

## 3. Project steps

| Step | What I did | Main output |
| :--- | :--- | :--- |
| 1 | Built a reusable structured prompt and tested it on obvious positive and negative examples | `review_prompt.py`, `spot_check.py` |
| 2 | Classified the first 100 reviews as POSITIVE or NEGATIVE | `step2_results.csv` |
| 3 | Built the first results dashboard | `step3_dashboard.html` |
| 4 | Added and verified interactive filtering | `verify_filters.py` |
| 5 | Compared LLM emotion detection with NRC word-list emotion detection | `step5_results.csv` |
| 6 | Created a balanced three-class sample and reran sentiment classification | `step6_results.csv` |
| 7 | Updated the dashboard for the balanced three-class results | `step7_dashboard.html` |

---

## 4. Step 2: Binary sentiment classification

For the first run, I classified the first 100 reviews using two sentiment classes.

The rating-based answer was defined as:

- 4–5 stars = **POSITIVE**
- 1–3 stars = **NEGATIVE**

The LLM never saw the rating.

### Results

| Metric | Result |
| :--- | ---: |
| Total reviews | 100 |
| True class mix | 93 POSITIVE / 7 NEGATIVE |
| Overall accuracy | **98.0%** |
| POSITIVE accuracy | 92/93 = **98.9%** |
| NEGATIVE accuracy | 6/7 = **85.7%** |

The star-rating mix was:

- 4 one-star reviews
- 1 two-star review
- 2 three-star reviews
- 2 four-star reviews
- 91 five-star reviews

Only two reviews did not match the rating-derived class.

One was a 5-star review that included a complaint about a missing note, so the text itself had mixed sentiment. The other was a 3-star review titled **"Easy to use"** that the model predicted as positive.

These mismatches showed that the wording of a review and its star rating do not always communicate the exact same sentiment.

### Why the first result looked so strong

The first 100 reviews were heavily skewed toward positive reviews. There were 93 positive reviews and only 7 negative reviews, including 91 five-star reviews.

Because of that, the **98.0% accuracy was strongly influenced by the class imbalance**. The model performed very well on positive reviews, and positive reviews made up almost the entire sample.

This is why the balanced sample in Step 6 was important.

---

## 5. Step 6: Balanced three-class classification

In Step 6, I changed the classification system to:

- 4–5 stars = **POSITIVE**
- 3 stars = **NEUTRAL**
- 1–2 stars = **NEGATIVE**

Instead of taking the first reviews in the dataset, I sampled **50 reviews from each class**, for a total of 150 reviews.

A fixed seed of `42` was used so the same reviews would be selected every time.

### Results

| Metric | Result |
| :--- | ---: |
| Total reviews | 150 |
| Class mix | 50 POSITIVE / 50 NEUTRAL / 50 NEGATIVE |
| Overall accuracy | **109/150 = 72.7%** |
| POSITIVE accuracy | 48/50 = **96.0%** |
| NEUTRAL accuracy | 13/50 = **26.0%** |
| NEGATIVE accuracy | 48/50 = **96.0%** |

### Confusion matrix

Rows show the true class and columns show the model prediction.

| True \ Predicted | POSITIVE | NEUTRAL | NEGATIVE |
| :--- | ---: | ---: | ---: |
| POSITIVE | 48 | 1 | 1 |
| NEUTRAL | 4 | 13 | **33** |
| NEGATIVE | 1 | 1 | 48 |

The biggest problem was the **NEUTRAL class**.

Out of 50 true neutral reviews:

- 13 were correctly classified as NEUTRAL
- 33 were classified as NEGATIVE
- 4 were classified as POSITIVE

This means the model handled clearly positive and clearly negative reviews very well, but struggled much more with reviews in the middle.

The balanced run gave a much different picture than the original binary run. Overall accuracy dropped from 98.0% to 72.7%, mainly because the balanced sample forced the model to handle the harder neutral class instead of mostly positive reviews.

One limitation of this particular balanced sample is that the 50 positive reviews happened to all be 5-star reviews. There were no 4-star reviews in the positive sample. The class balance is still correct, but the positive class has less star-rating variety than I would ideally want.

---

## 6. Step 5: LLM emotion vs. NRC emotion

I also compared two ways of assigning a primary emotion to each review.

### LLM method

The prompt asked the model to return one primary emotion from:

- anger
- anticipation
- disgust
- fear
- joy
- sadness
- surprise
- trust

### NRC word-list method

The review words were compared with the NRC Emotion Lexicon.

Each matching word contributed to one or more emotion scores. The scores were added together and the emotion with the highest score became the NRC prediction.

If no words matched the lexicon, the NRC result was recorded as `none`.

### Results

The two methods agreed on:

**20/100 reviews = 20.0%**

| Emotion | LLM | NRC |
| :--- | ---: | ---: |
| joy | 87 | 21 |
| anticipation | 3 | 59 |
| anger | 6 | 2 |
| trust | 2 | 1 |
| surprise | 1 | 0 |
| sadness | 1 | 1 |
| disgust | 0 | 1 |
| fear | 0 | 0 |
| none | N/A | 15 |

The LLM predicted **joy** most often, while the NRC method predicted **anticipation** most often.

A major reason for the difference was the way the NRC method handles tied scores. Gift-card words such as "gift," "great," "good," and "easy" often produced ties between multiple emotions. Because the code used a fixed tie-breaking order, anticipation was often selected.

The NRC approach also had trouble when a review expressed frustration without using a word that was directly associated with that emotion in the lexicon.

For example:

> "Card did not work!!!!"

The LLM classified the emotion as anger, while the NRC method did not have enough direct word-level information to interpret the full meaning in the same way.

Overall, the LLM was able to use the context of the full review, while the NRC method was limited to the emotion associations of individual words.

---

## 7. Issues I ran into

A few issues came up while completing the project.

### Dataset link

The dataset link from the assignment returned a 404 error.

I worked around this by using the official McAuley Lab Hugging Face mirror and confirming that the dataset contained 152,410 reviews.

### Model configuration

The endpoint did not accept the original model name that was first tried. I checked the models available on the endpoint and used the model ID that the server actually provided.

### NRC Emotion Lexicon

Several NRC lexicon download links did not work. I eventually used the official NRC Emotion Lexicon zip file and also kept a JSON fallback copy.

### NRC scoring bug

A test found that the `primary_emotion()` function could fail when some emotion keys were missing. I fixed this by safely defaulting missing scores to zero.

### Dashboard testing

The dashboard displayed correctly in the preview pane, but automated clicks did not work reliably there.

Because of that, I used Playwright with a real browser engine to verify that the filters returned the correct rows and that the live counts matched the CSV results.

### Sampling and imbalance

One of the biggest lessons from the project was how much the sample can affect the results.

The first 100 reviews produced 98.0% accuracy, but they were overwhelmingly positive. Once I used a balanced sample and added the neutral class, overall accuracy dropped to 72.7%.

This made it much easier to see what the model was actually good at and where it struggled.

---

## 8. Project files and reproduction

### Main project files

- `review_prompt.py`
- `spot_check.py`
- `step2_classify.py`
- `step3_dashboard.py`
- `verify_filters.py`
- `step5_emotions.py`
- `step6_balanced.py`
- `step7_dashboard.py`
- `verify_step7.py`

### Saved outputs

- `step2_results.csv`
- `step5_results.csv`
- `step6_sample.csv`
- `step6_results.csv`
- `step3_dashboard.html`
- `step7_dashboard.html`

The `.env` file stores the API information locally and is excluded by `.gitignore`.

### Reproduce the project

```bash
# Step 1
python spot_check.py

# Step 2
python step2_classify.py

# Step 3
python step3_dashboard.py

# Step 5
python step5_emotions.py

# Step 6
python step6_balanced.py --k 50 --seed 42

# Step 7
python step7_dashboard.py

```

## 9. References

- Hou, Y. et al. (2024). *Bridging Language and Items for Retrieval and Recommendation*. Amazon Reviews '23, McAuley Lab.
- Amazon Reviews '23 project page: <https://amazon-reviews-2023.github.io/>
- Amazon Reviews '23 Hugging Face mirror: <https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023>
- Mohammad, S. M., & Turney, P. D. (2013). "Crowdsourcing a Word-Emotion Association Lexicon." *Computational Intelligence, 29*(3), 436–465.
- NRC Emotion Lexicon: <https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm>

---

## 10. Screenshots

### Step 7 Dashboard Overview

![Step 7 dashboard overview](<screenshots/Full Step 7 dashboard overview.png>)

### True Class vs. Model Prediction

![True class vs model prediction](<screenshots/True class vs. model-predicted class.png>)

### Accuracy by Class and Neutral Results

![Accuracy by class and neutral results](<screenshots/Accuracy by class + Neutral problem.png>)

### Confusion Matrix

![Confusion matrix](<screenshots/Confusion Matrix.png>)