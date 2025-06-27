# Credit Risk Model

This project implements an end-to-end Credit Scoring Model for alternative data sourced from an eCommerce platform. It is designed to assess customer creditworthiness and support a Buy-Now-Pay-Later service for Bati Bank.

## Tilte: Credit Scoring Business Understanding

# 1. Basel II’s Influence on Model Interpretability

The Basel II Capital Accord emphasizes the importance of risk measurement and regulatory compliance for financial institutions. It mandates that banks must not only quantify credit risk but also justify and document the process used. As such, interpretability is essential—regulators must be able to understand how the model arrives at its predictions. Models that offer transparency, such as logistic regression with clear variable contributions (e.g., via Weight of Evidence), align better with Basel II expectations. Interpretability enhances trust, ensures compliance, and helps detect potential bias or unfair decision-making.

# 2. Why a Proxy Label is Needed

The dataset lacks a direct indicator of whether a customer defaulted on a loan. In the absence of this target variable, a proxy is required to simulate risk behavior. This proxy is created by segmenting customers based on Recency, Frequency, and Monetary (RFM) transaction behaviors. Those with low engagement (e.g., infrequent purchases, low spend) are treated as higher risk. This approach enables model training but introduces business risks—if the proxy poorly approximates actual defaults, model predictions may misclassify customers, either rejecting creditworthy individuals or approving risky ones. Proper validation and domain input are necessary to reduce this risk.

# 3. Trade-offs: Simple vs. Complex Models in Finance

In regulated financial environments, there is a trade-off between model simplicity and performance. Simple models like logistic regression are easy to interpret, audit, and explain—crucial for transparency and regulatory approval. They are also faster to train and deploy. However, they may not capture complex patterns in data, leading to suboptimal performance.

Complex models such as Gradient Boosting (e.g., XGBoost, LightGBM) often achieve higher predictive accuracy by modeling nonlinear interactions. But they are harder to interpret, require more computation, and may raise concerns during audits. Model explainability techniques (e.g., SHAP values) can mitigate this but do not replace the clarity of simpler models. The final choice depends on balancing performance needs with regulatory constraints and business goals.
