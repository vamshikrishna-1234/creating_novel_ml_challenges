# Initial Rubrics for Code-Switched Resolution Hours Challenge

Aim for 5+ rubrics; majority REQUIRED or RECOMMENDED; task-specific (not generic). Use these as the initial rubric set when creating the challenge.

---

1. **Type:** DATA_HANDLING | **Importance:** REQUIRED  
   **Criterion:** Loads and uses both `text` and `category` from train/test (no dropping category).  
   **Rationale:** Category is a strong signal for resolution time; ignoring it leaves performance on the table and is effectively broken for this task.

2. **Type:** FEATURE_ENGINEERING | **Importance:** RECOMMENDED  
   **Criterion:** Uses at least one text-derived feature beyond raw bag-of-words (e.g. length, code-switch ratio, presence of priority tags [P1]–[P5], or embeddings).  
   **Rationale:** The target depends on code-switch density and embedded tags; models that only use naive BoW will plateau.

3. **Type:** TRAINING | **Importance:** RECOMMENDED  
   **Criterion:** Uses a proper train/validation split or cross-validation for model selection or early stopping; does not tune or select models using the test set.  
   **Rationale:** Prevents overfitting and test leakage; task-specific because resolution-time distribution is wide and validation strategy matters.

4. **Type:** MODELING | **Importance:** RECOMMENDED  
   **Criterion:** Uses a regression formulation (predicts continuous hours) and does not treat the task as classification (e.g. binning into classes only).  
   **Rationale:** Evaluation is RMSE on continuous values; discretizing to few classes loses information and hurts score.

5. **Type:** CODE_QUALITY | **Importance:** REQUIRED  
   **Criterion:** Submission CSV has exactly columns `id` and `prediction`, with one row per test id and no duplicate ids.  
   **Rationale:** Grader expects this format; wrong format causes grading failure (broken solution).

6. **Type:** DATA_HANDLING | **Importance:** RECOMMENDED  
   **Criterion:** Handles text encoding (UTF-8) and does not drop or corrupt rows with special characters or Verani tokens.  
   **Rationale:** Code-switched text contains non-ASCII tokens; broken encoding leads to missing or wrong features.

7. **Type:** UNIVERSAL | **Importance:** UNIVERSAL  
   **Criterion:** Does not use test set (or test labels) for training, feature computation, or normalization.  
   **Rationale:** Universal anti-leakage criterion; applies to any supervised challenge.

---

**Summary:** 5 RECOMMENDED, 2 REQUIRED, 1 UNIVERSAL. All are verifiable and task-specific except the one UNIVERSAL.
