# 🧠 Behavioral Personality Classifier

> Can a machine understand your personality better than you think?
> 
> This project takes 29 behavioral traits — how social you are, how you handle stress, how curious you get — and predicts your personality type using machine learning.

---

## 📌 What This Project Does

Most personality tests feel like a questionnaire that gives you a label.

This project goes one step further — it **compares 4 different ML models** to find which one understands human behavior the best, cleans messy data the smart way, and finally lets **you** enter your own traits and get your personality predicted live.

It's not just a model. It's a small system that thinks about people. 🤖

---

## 📦 Dataset

| Property | Detail |
|---|---|
| Rows | 20,000 |
| Features | 29 behavioral traits |
| Target | Personality Type |
| Source | Synthetic behavioral dataset |

**Some of the features:**
`social_energy`, `alone_time_preference`, `leadership`, `empathy`, `creativity`, `stress_handling`, `decision_speed`, `adventurousness` — and 21 more.

---

## 🧹 Data Cleaning — The Smart Way

Before touching any model, the data was cleaned properly:

- Checked for NULL values, duplicates, and zero values
- Replaced 0s with `NaN` (because 0 on a behavioral scale doesn't make sense)
- **Skewness-based imputation** — if a feature was normally distributed (skew between -1 and 1), filled with **mean**. Otherwise filled with **median**. This matters because blindly filling everything with mean on skewed data gives wrong insights.

```python
skw = personality_Data[col].skew()
if (skw > -1 and skw < 1):
    personality_Data[col] = personality_Data[col].fillna(personality_Data[col].mean())
else:
    personality_Data[col] = personality_Data[col].fillna(personality_Data[col].median())
```

---

## ⚙️ ML Pipeline

```
Raw Data → Cleaning → Label Encoding → Train-Test Split → Model Training → Evaluation → Prediction
```

**Why no scaling?**
Tree-based models (Random Forest) and distance-based models like KNN on this dataset didn't require feature scaling after testing — so it was intentionally skipped.

---

## 🤖 Models Compared

| Model | Accuracy |
|---|---|
| Logistic Regression | 99.70% |
| KNN (tuned) | **99.75%** ✅ |
| Random Forest | 99.35% |
| SVM (linear / rbf / poly) | ~99.70% |

**KNN was selected as the final model** after hyperparameter tuning across k=1 to k=10.

```python
acc_sc_list = []
for i in range(1, 11):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(X_train, y_train)
    acc_sc_list.append(accuracy_score(y_test, knn_model.predict(X_test)))

best_k = acc_sc_list.index(max(acc_sc_list))
```

---

## 📊 Visualizations

- **Class Distribution** — are personality types balanced in the dataset?
- **Correlation Heatmap** — which behavioral traits move together?
- **Box Plots** — how do traits like `social_energy`, `leadership`, `empathy` differ across personality types?
- **Feature Importances** — which traits matter most according to Random Forest?
- **Confusion Matrix** — where does the best model (KNN) get confused?

---

## 🎯 Live User Prediction

The notebook ends with a real interaction — you rate yourself on all 29 traits from 1 to 10, and the model tells you your personality type.

```
social_energy (1-10) : 8
alone_time_preference (1-10) : 3
leadership (1-10) : 7
...

🎯 Your Personality Type : Extrovert
```

Because what's the point of a personality classifier if you can't try it yourself? ✨

---

## 🛠️ Tech Stack

| Tool | Use |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data handling and cleaning |
| Scikit-learn | ML models, encoding, splitting |
| Matplotlib & Seaborn | Visualizations |
| Google Colab | Development environment |

---

## 📁 Files

| File | Description |
|---|---|
| `behavioral_personality_classifier.ipynb` | Complete notebook — cleaning, models, visualizations, user prediction |
| `personality_synthetic_dataset.csv` | Dataset (20,000 rows, 29 features) |

---

## 💭 What I Learned

This project taught me that data cleaning is not just removing nulls — it's **understanding the data's nature** before deciding how to fix it. Skewness-based imputation was a small decision that made a real difference.

And comparing 4 models side by side showed me something important — accuracy alone doesn't tell the full story. Understanding *why* one model performs better than another is where the real learning happens.

> Sometimes the most interesting personality in the room... is the data itself. 🧠

---

*Built with Python · Scikit-learn · Part of my ML portfolio*
